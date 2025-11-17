module @jit_train_step attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<i32> {mhlo.sharding = "{replicated}", tf.aliasing_output = 0 : i32}, %arg1: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 1 : i32}, %arg2: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 2 : i32}, %arg3: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 3 : i32}, %arg4: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 4 : i32}, %arg5: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 5 : i32}, %arg6: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 6 : i32}, %arg7: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 7 : i32}, %arg8: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 8 : i32}, %arg9: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 9 : i32}, %arg10: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 10 : i32}, %arg11: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 11 : i32}, %arg12: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 12 : i32}, %arg13: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 13 : i32}, %arg14: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 14 : i32}, %arg15: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 15 : i32}, %arg16: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 16 : i32}, %arg17: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 17 : i32}, %arg18: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 18 : i32}, %arg19: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 19 : i32}, %arg20: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 20 : i32}, %arg21: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 21 : i32}, %arg22: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 22 : i32}, %arg23: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 23 : i32}, %arg24: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 24 : i32}, %arg25: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 25 : i32}, %arg26: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 26 : i32}, %arg27: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 27 : i32}, %arg28: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 28 : i32}, %arg29: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 29 : i32}, %arg30: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 30 : i32}, %arg31: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 31 : i32}, %arg32: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 32 : i32}, %arg33: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 33 : i32}, %arg34: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 34 : i32}, %arg35: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 35 : i32}, %arg36: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 36 : i32}, %arg37: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 37 : i32}, %arg38: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 38 : i32}, %arg39: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 39 : i32}, %arg40: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 40 : i32}, %arg41: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 41 : i32}, %arg42: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 42 : i32}, %arg43: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 43 : i32}, %arg44: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 44 : i32}, %arg45: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 45 : i32}, %arg46: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 46 : i32}, %arg47: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 47 : i32}, %arg48: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 48 : i32}, %arg49: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 49 : i32}, %arg50: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 50 : i32}, %arg51: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 51 : i32}, %arg52: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 52 : i32}, %arg53: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 53 : i32}, %arg54: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 54 : i32}, %arg55: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 55 : i32}, %arg56: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 56 : i32}, %arg57: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 57 : i32}, %arg58: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 58 : i32}, %arg59: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 59 : i32}, %arg60: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 60 : i32}, %arg61: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 61 : i32}, %arg62: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 62 : i32}, %arg63: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 63 : i32}, %arg64: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 64 : i32}, %arg65: tensor<576x128256xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 65 : i32}, %arg66: tensor<128256x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 66 : i32}, %arg67: tensor<i32> {mhlo.sharding = "{replicated}", tf.aliasing_output = 67 : i32}, %arg68: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 68 : i32}, %arg69: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 69 : i32}, %arg70: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 70 : i32}, %arg71: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 71 : i32}, %arg72: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 72 : i32}, %arg73: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 73 : i32}, %arg74: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 74 : i32}, %arg75: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 75 : i32}, %arg76: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 76 : i32}, %arg77: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 77 : i32}, %arg78: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 78 : i32}, %arg79: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 79 : i32}, %arg80: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 80 : i32}, %arg81: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 81 : i32}, %arg82: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 82 : i32}, %arg83: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 83 : i32}, %arg84: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 84 : i32}, %arg85: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 85 : i32}, %arg86: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 86 : i32}, %arg87: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 87 : i32}, %arg88: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 88 : i32}, %arg89: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 89 : i32}, %arg90: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 90 : i32}, %arg91: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 91 : i32}, %arg92: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 92 : i32}, %arg93: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 93 : i32}, %arg94: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 94 : i32}, %arg95: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 95 : i32}, %arg96: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 96 : i32}, %arg97: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 97 : i32}, %arg98: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 98 : i32}, %arg99: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 99 : i32}, %arg100: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 100 : i32}, %arg101: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 101 : i32}, %arg102: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 102 : i32}, %arg103: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 103 : i32}, %arg104: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 104 : i32}, %arg105: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 105 : i32}, %arg106: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 106 : i32}, %arg107: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 107 : i32}, %arg108: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 108 : i32}, %arg109: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 109 : i32}, %arg110: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 110 : i32}, %arg111: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 111 : i32}, %arg112: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 112 : i32}, %arg113: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 113 : i32}, %arg114: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 114 : i32}, %arg115: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 115 : i32}, %arg116: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 116 : i32}, %arg117: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 117 : i32}, %arg118: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 118 : i32}, %arg119: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 119 : i32}, %arg120: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 120 : i32}, %arg121: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 121 : i32}, %arg122: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 122 : i32}, %arg123: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 123 : i32}, %arg124: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 124 : i32}, %arg125: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 125 : i32}, %arg126: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 126 : i32}, %arg127: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 127 : i32}, %arg128: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 128 : i32}, %arg129: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 129 : i32}, %arg130: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 130 : i32}, %arg131: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 131 : i32}, %arg132: tensor<576x128256xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 132 : i32}, %arg133: tensor<128256x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 133 : i32}, %arg134: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 134 : i32}, %arg135: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 135 : i32}, %arg136: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 136 : i32}, %arg137: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 137 : i32}, %arg138: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 138 : i32}, %arg139: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 139 : i32}, %arg140: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 140 : i32}, %arg141: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 141 : i32}, %arg142: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 142 : i32}, %arg143: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 143 : i32}, %arg144: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 144 : i32}, %arg145: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 145 : i32}, %arg146: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 146 : i32}, %arg147: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 147 : i32}, %arg148: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 148 : i32}, %arg149: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 149 : i32}, %arg150: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 150 : i32}, %arg151: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 151 : i32}, %arg152: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 152 : i32}, %arg153: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 153 : i32}, %arg154: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 154 : i32}, %arg155: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 155 : i32}, %arg156: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 156 : i32}, %arg157: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 157 : i32}, %arg158: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 158 : i32}, %arg159: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 159 : i32}, %arg160: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 160 : i32}, %arg161: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 161 : i32}, %arg162: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 162 : i32}, %arg163: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 163 : i32}, %arg164: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 164 : i32}, %arg165: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 165 : i32}, %arg166: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 166 : i32}, %arg167: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 167 : i32}, %arg168: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 168 : i32}, %arg169: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 169 : i32}, %arg170: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 170 : i32}, %arg171: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 171 : i32}, %arg172: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 172 : i32}, %arg173: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 173 : i32}, %arg174: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 174 : i32}, %arg175: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 175 : i32}, %arg176: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 176 : i32}, %arg177: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 177 : i32}, %arg178: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 178 : i32}, %arg179: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 179 : i32}, %arg180: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 180 : i32}, %arg181: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 181 : i32}, %arg182: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 182 : i32}, %arg183: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 183 : i32}, %arg184: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 184 : i32}, %arg185: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 185 : i32}, %arg186: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 186 : i32}, %arg187: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 187 : i32}, %arg188: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 188 : i32}, %arg189: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 189 : i32}, %arg190: tensor<576x2016xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 190 : i32}, %arg191: tensor<2016x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 191 : i32}, %arg192: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 192 : i32}, %arg193: tensor<576xbf16> {mhlo.sharding = "{replicated}", tf.aliasing_output = 193 : i32}, %arg194: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 194 : i32}, %arg195: tensor<9x128x576xbf16> {mhlo.sharding = "{devices=[1,1,8]<=[8]}", tf.aliasing_output = 195 : i32}, %arg196: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 196 : i32}, %arg197: tensor<576x9x128xbf16> {mhlo.sharding = "{devices=[8,1,1]<=[8]}", tf.aliasing_output = 197 : i32}, %arg198: tensor<576x128256xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}", tf.aliasing_output = 198 : i32}, %arg199: tensor<128256x576xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}", tf.aliasing_output = 199 : i32}, %arg200: tensor<i32> {mhlo.sharding = "{replicated}", tf.aliasing_output = 200 : i32}, %arg201: tensor<8x4096xi32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg202: tensor<8x4096xi32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg203: tensor<8x4096xi32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg204: tensor<8x4096xi32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg205: tensor<8x4096xi32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}) -> (tensor<i32> {jax.result_info = "result[0].step", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['decoder_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_0']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_0']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_0']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_0']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_0']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_0']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_0']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_0']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_0']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_1']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_1']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_1']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_1']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_1']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_1']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_1']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_1']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_1']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_2']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_2']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_2']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_2']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_2']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_2']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_2']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_2']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_2']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_3']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_3']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_3']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_3']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_3']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_3']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_3']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_3']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_3']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_4']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_4']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_4']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_4']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_4']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_4']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_4']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_4']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_4']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_5']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_5']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_5']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_5']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_5']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_5']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_5']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_5']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_5']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_6']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_6']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_6']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_6']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_6']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_6']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_6']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_6']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].params['params']['decoder']['layers_6']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x128256xbf16> {jax.result_info = "result[0].params['params']['decoder']['logits_dense']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<128256x576xbf16> {jax.result_info = "result[0].params['params']['token_embedder']['embedding']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<i32> {jax.result_info = "result[0].opt_state[0].count", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['decoder_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_0']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_0']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_0']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_0']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_0']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_0']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_0']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_0']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_0']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_1']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_1']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_1']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_1']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_1']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_1']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_1']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_1']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_1']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_2']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_2']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_2']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_2']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_2']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_2']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_2']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_2']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_2']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_3']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_3']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_3']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_3']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_3']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_3']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_3']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_3']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_3']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_4']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_4']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_4']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_4']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_4']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_4']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_4']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_4']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_4']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_5']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_5']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_5']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_5']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_5']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_5']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_5']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_5']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_5']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_6']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_6']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_6']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_6']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_6']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_6']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_6']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_6']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['layers_6']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x128256xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['decoder']['logits_dense']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<128256x576xbf16> {jax.result_info = "result[0].opt_state[0].mu['params']['token_embedder']['embedding']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['decoder_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_0']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_0']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_0']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_0']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_0']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_0']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_0']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_0']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_0']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_1']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_1']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_1']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_1']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_1']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_1']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_1']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_1']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_1']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_2']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_2']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_2']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_2']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_2']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_2']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_2']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_2']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_2']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_3']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_3']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_3']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_3']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_3']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_3']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_3']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_3']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_3']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_4']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_4']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_4']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_4']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_4']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_4']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_4']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_4']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_4']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_5']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_5']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_5']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_5']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_5']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_5']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_5']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_5']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_5']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_6']['mlp']['wi_0']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<576x2016xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_6']['mlp']['wi_1']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<2016x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_6']['mlp']['wo']['kernel']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_6']['post_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_6']['pre_self_attention_layer_norm']['scale']", mhlo.sharding = "{replicated}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_6']['self_attention']['key']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<9x128x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_6']['self_attention']['out']['kernel']", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_6']['self_attention']['query']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x9x128xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['layers_6']['self_attention']['value']['kernel']", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}, tensor<576x128256xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['decoder']['logits_dense']['kernel']", mhlo.sharding = "{devices=[8,1]<=[8]}"}, tensor<128256x576xbf16> {jax.result_info = "result[0].opt_state[0].nu['params']['token_embedder']['embedding']", mhlo.sharding = "{devices=[1,8]<=[8]}"}, tensor<i32> {jax.result_info = "result[0].opt_state[2].count", mhlo.sharding = "{replicated}"}, tensor<bf16> {jax.result_info = "result[1]['scalar']['learning/grad_norm']"}, tensor<f32> {jax.result_info = "result[1]['scalar']['learning/loss']"}, tensor<f32> {jax.result_info = "result[1]['scalar']['learning/moe_lb_loss']"}, tensor<f32> {jax.result_info = "result[1]['scalar']['learning/mtp_loss']"}, tensor<bf16> {jax.result_info = "result[1]['scalar']['learning/param_norm']"}, tensor<bf16> {jax.result_info = "result[1]['scalar']['learning/raw_grad_norm']"}, tensor<i32> {jax.result_info = "result[1]['scalar']['learning/total_weights']"}) {
    %0 = stablehlo.iota dim = 0 : tensor<128256xi32>
    %1 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [2] : (tensor<128256xi32>) -> tensor<1x1x128256xi32>
    %3 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x128256xi32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<1x1x128256xi32>) -> tensor<8x4096x128256xi32>
    %5 = stablehlo.compare  EQ, %3, %4,  SIGNED : (tensor<8x4096x128256xi32>, tensor<8x4096x128256xi32>) -> tensor<8x4096x128256xi1>
    %6 = stablehlo.convert %5 : (tensor<8x4096x128256xi1>) -> tensor<8x4096x128256xbf16>
    %7 = stablehlo.dot_general %6, %arg66, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x128256xbf16>, tensor<128256x576xbf16>) -> tensor<8x4096x576xbf16>
    %8 = stablehlo.custom_call @Sharding(%7) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %9 = stablehlo.custom_call @Sharding(%8) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %10 = stablehlo.convert %9 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<8x4096x576xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_0 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %15 = stablehlo.divide %13, %14 : tensor<8x4096x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %17 = stablehlo.add %15, %16 : tensor<8x4096x1xf32>
    %18 = stablehlo.rsqrt %17 : tensor<8x4096x1xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %20 = stablehlo.multiply %10, %19 : tensor<8x4096x576xf32>
    %21 = stablehlo.convert %20 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %22 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %24 = stablehlo.multiply %21, %23 : tensor<8x4096x576xbf16>
    %25 = stablehlo.custom_call @Sharding(%24) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %26 = stablehlo.custom_call @Sharding(%25) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %27 = stablehlo.custom_call @Sharding(%25) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %28 = stablehlo.dot_general %26, %arg9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %29 = stablehlo.dot_general %27, %arg7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %30 = stablehlo.dot_general %27, %arg10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %31 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %32 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %33 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %34 = stablehlo.multiply %33, %32 : tensor<64xi32>
    %35 = stablehlo.convert %34 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_2 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %37 = stablehlo.divide %35, %36 : tensor<64xf32>
    %cst_3 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %39 = stablehlo.power %38, %37 : tensor<64xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<64xf32>
    %42 = stablehlo.convert %31 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %43 = stablehlo.broadcast_in_dim %41, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %44 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %45 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %46 = stablehlo.divide %44, %45 : tensor<8x4096x1x64xf32>
    %47 = stablehlo.sine %46 : tensor<8x4096x1x64xf32>
    %48 = stablehlo.convert %47 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %49 = stablehlo.cosine %46 : tensor<8x4096x1x64xf32>
    %50 = stablehlo.convert %49 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %51 = stablehlo.slice %28 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %52 = stablehlo.slice %28 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %53 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %54 = stablehlo.multiply %51, %53 : tensor<8x4096x9x64xbf16>
    %55 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %56 = stablehlo.multiply %52, %55 : tensor<8x4096x9x64xbf16>
    %57 = stablehlo.subtract %54, %56 : tensor<8x4096x9x64xbf16>
    %58 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %59 = stablehlo.multiply %52, %58 : tensor<8x4096x9x64xbf16>
    %60 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %61 = stablehlo.multiply %51, %60 : tensor<8x4096x9x64xbf16>
    %62 = stablehlo.add %59, %61 : tensor<8x4096x9x64xbf16>
    %63 = stablehlo.concatenate %57, %62, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %64 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %65 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_5 = stablehlo.constant dense<2> : tensor<i32>
    %66 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %67 = stablehlo.multiply %66, %65 : tensor<64xi32>
    %68 = stablehlo.convert %67 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_6 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %69 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %70 = stablehlo.divide %68, %69 : tensor<64xf32>
    %cst_7 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %71 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %72 = stablehlo.power %71, %70 : tensor<64xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %74 = stablehlo.multiply %73, %72 : tensor<64xf32>
    %75 = stablehlo.convert %64 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %76 = stablehlo.broadcast_in_dim %74, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %77 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %78 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %79 = stablehlo.divide %77, %78 : tensor<8x4096x1x64xf32>
    %80 = stablehlo.sine %79 : tensor<8x4096x1x64xf32>
    %81 = stablehlo.convert %80 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %82 = stablehlo.cosine %79 : tensor<8x4096x1x64xf32>
    %83 = stablehlo.convert %82 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %84 = stablehlo.slice %29 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %85 = stablehlo.slice %29 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %86 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %87 = stablehlo.multiply %84, %86 : tensor<8x4096x9x64xbf16>
    %88 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %89 = stablehlo.multiply %85, %88 : tensor<8x4096x9x64xbf16>
    %90 = stablehlo.subtract %87, %89 : tensor<8x4096x9x64xbf16>
    %91 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %92 = stablehlo.multiply %85, %91 : tensor<8x4096x9x64xbf16>
    %93 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %94 = stablehlo.multiply %84, %93 : tensor<8x4096x9x64xbf16>
    %95 = stablehlo.add %92, %94 : tensor<8x4096x9x64xbf16>
    %96 = stablehlo.concatenate %90, %95, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %97 = stablehlo.custom_call @Sharding(%63) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %98 = stablehlo.custom_call @Sharding(%96) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %99 = stablehlo.custom_call @Sharding(%30) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %100 = stablehlo.reduce_precision %97, format = e8m7 : tensor<8x4096x9x128xbf16>
    %101 = stablehlo.reduce_precision %98, format = e8m7 : tensor<8x4096x9x128xbf16>
    %102 = stablehlo.reduce_precision %99, format = e8m7 : tensor<8x4096x9x128xbf16>
    %103 = stablehlo.reshape %100 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %104 = stablehlo.dot_general %101, %103, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %105 = stablehlo.transpose %104, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %106 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %107 = stablehlo.broadcast_in_dim %arg203, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %108 = stablehlo.broadcast_in_dim %106, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %109 = stablehlo.broadcast_in_dim %107, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %110 = stablehlo.compare  EQ, %108, %109,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %112 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %113 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_9 = stablehlo.constant dense<0> : tensor<i32>
    %114 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %115 = stablehlo.add %112, %114 : tensor<4096x4096xi32>
    %116 = stablehlo.compare  LE, %113, %115,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %117 = stablehlo.broadcast_in_dim %116, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %118 = stablehlo.broadcast_in_dim %117, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %119 = stablehlo.and %111, %118 : tensor<8x1x1x4096x4096xi1>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_11 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %120 = call @_where(%119, %cst_10, %cst_11) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_12 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %121 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %122 = stablehlo.compare  GE, %120, %121,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_13 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %123 = call @_where_34(%122, %105, %cst_13) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> tensor<8x9x1x4096x4096xbf16>
    %124 = stablehlo.reshape %123 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_14 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %125 = stablehlo.reduce(%124 init: %cst_14) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %126 = stablehlo.broadcast_in_dim %125, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %127 = stablehlo.broadcast_in_dim %126, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %128 = stablehlo.subtract %124, %127 : tensor<8x9x4096x4096xbf16>
    %129 = stablehlo.exponential %128 : tensor<8x9x4096x4096xbf16>
    %130 = stablehlo.convert %129 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %131 = stablehlo.reduce(%130 init: %cst_15) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %132 = stablehlo.broadcast_in_dim %131, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %133 = stablehlo.convert %132 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %134 = stablehlo.reshape %129 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %135 = stablehlo.transpose %133, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %136 = stablehlo.dot_general %102, %134, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %137 = stablehlo.transpose %136, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %138 = stablehlo.reshape %137 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %139 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %140 = stablehlo.divide %138, %139 : tensor<8x4096x9x128xbf16>
    %141 = stablehlo.custom_call @Sharding(%140) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %142 = stablehlo.dot_general %141, %arg8, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x576xbf16>
    %143 = stablehlo.reduce_precision %142, format = e8m7 : tensor<8x4096x576xbf16>
    %144 = stablehlo.custom_call @Sharding(%143) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %145 = stablehlo.add %9, %144 : tensor<8x4096x576xbf16>
    %146 = stablehlo.convert %145 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %147 = stablehlo.multiply %146, %146 : tensor<8x4096x576xf32>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %148 = stablehlo.reduce(%147 init: %cst_16) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_17 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %150 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %151 = stablehlo.divide %149, %150 : tensor<8x4096x1xf32>
    %cst_18 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %152 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %153 = stablehlo.add %151, %152 : tensor<8x4096x1xf32>
    %154 = stablehlo.rsqrt %153 : tensor<8x4096x1xf32>
    %155 = stablehlo.broadcast_in_dim %154, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %156 = stablehlo.multiply %146, %155 : tensor<8x4096x576xf32>
    %157 = stablehlo.convert %156 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %158 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %159 = stablehlo.broadcast_in_dim %158, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %160 = stablehlo.multiply %157, %159 : tensor<8x4096x576xbf16>
    %161 = stablehlo.custom_call @Sharding(%160) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %162 = stablehlo.dot_general %161, %arg2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %163:2 = call @silu(%162) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %164 = stablehlo.dot_general %161, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %165 = stablehlo.reduce_precision %164, format = e8m7 : tensor<8x4096x2016xbf16>
    %166 = stablehlo.multiply %163#0, %165 : tensor<8x4096x2016xbf16>
    %167 = stablehlo.custom_call @Sharding(%166) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %168 = stablehlo.dot_general %167, %arg4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x576xbf16>
    %169 = stablehlo.custom_call @Sharding(%168) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %170 = stablehlo.add %169, %145 : tensor<8x4096x576xbf16>
    %171 = stablehlo.custom_call @Sharding(%170) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %172 = stablehlo.custom_call @Sharding(%171) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %173 = stablehlo.convert %172 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %174 = stablehlo.multiply %173, %173 : tensor<8x4096x576xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %175 = stablehlo.reduce(%174 init: %cst_19) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %176 = stablehlo.broadcast_in_dim %175, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_20 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %177 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %178 = stablehlo.divide %176, %177 : tensor<8x4096x1xf32>
    %cst_21 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %179 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %180 = stablehlo.add %178, %179 : tensor<8x4096x1xf32>
    %181 = stablehlo.rsqrt %180 : tensor<8x4096x1xf32>
    %182 = stablehlo.broadcast_in_dim %181, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %183 = stablehlo.multiply %173, %182 : tensor<8x4096x576xf32>
    %184 = stablehlo.convert %183 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %185 = stablehlo.broadcast_in_dim %arg15, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %186 = stablehlo.broadcast_in_dim %185, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %187 = stablehlo.multiply %184, %186 : tensor<8x4096x576xbf16>
    %188 = stablehlo.custom_call @Sharding(%187) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %189 = stablehlo.custom_call @Sharding(%188) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %190 = stablehlo.custom_call @Sharding(%188) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %191 = stablehlo.dot_general %189, %arg18, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %192 = stablehlo.dot_general %190, %arg16, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %193 = stablehlo.dot_general %190, %arg19, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %194 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %195 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_22 = stablehlo.constant dense<2> : tensor<i32>
    %196 = stablehlo.broadcast_in_dim %c_22, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %197 = stablehlo.multiply %196, %195 : tensor<64xi32>
    %198 = stablehlo.convert %197 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_23 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %199 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %200 = stablehlo.divide %198, %199 : tensor<64xf32>
    %cst_24 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %201 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %202 = stablehlo.power %201, %200 : tensor<64xf32>
    %cst_25 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %203 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %204 = stablehlo.multiply %203, %202 : tensor<64xf32>
    %205 = stablehlo.convert %194 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %206 = stablehlo.broadcast_in_dim %204, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %207 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %208 = stablehlo.broadcast_in_dim %206, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %209 = stablehlo.divide %207, %208 : tensor<8x4096x1x64xf32>
    %210 = stablehlo.sine %209 : tensor<8x4096x1x64xf32>
    %211 = stablehlo.convert %210 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %212 = stablehlo.cosine %209 : tensor<8x4096x1x64xf32>
    %213 = stablehlo.convert %212 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %214 = stablehlo.slice %191 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %215 = stablehlo.slice %191 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %216 = stablehlo.broadcast_in_dim %213, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %217 = stablehlo.multiply %214, %216 : tensor<8x4096x9x64xbf16>
    %218 = stablehlo.broadcast_in_dim %211, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %219 = stablehlo.multiply %215, %218 : tensor<8x4096x9x64xbf16>
    %220 = stablehlo.subtract %217, %219 : tensor<8x4096x9x64xbf16>
    %221 = stablehlo.broadcast_in_dim %213, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %222 = stablehlo.multiply %215, %221 : tensor<8x4096x9x64xbf16>
    %223 = stablehlo.broadcast_in_dim %211, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %224 = stablehlo.multiply %214, %223 : tensor<8x4096x9x64xbf16>
    %225 = stablehlo.add %222, %224 : tensor<8x4096x9x64xbf16>
    %226 = stablehlo.concatenate %220, %225, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %227 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %228 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_26 = stablehlo.constant dense<2> : tensor<i32>
    %229 = stablehlo.broadcast_in_dim %c_26, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %230 = stablehlo.multiply %229, %228 : tensor<64xi32>
    %231 = stablehlo.convert %230 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_27 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %232 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %233 = stablehlo.divide %231, %232 : tensor<64xf32>
    %cst_28 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %234 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %235 = stablehlo.power %234, %233 : tensor<64xf32>
    %cst_29 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %236 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %237 = stablehlo.multiply %236, %235 : tensor<64xf32>
    %238 = stablehlo.convert %227 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %239 = stablehlo.broadcast_in_dim %237, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %240 = stablehlo.broadcast_in_dim %238, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %241 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %242 = stablehlo.divide %240, %241 : tensor<8x4096x1x64xf32>
    %243 = stablehlo.sine %242 : tensor<8x4096x1x64xf32>
    %244 = stablehlo.convert %243 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %245 = stablehlo.cosine %242 : tensor<8x4096x1x64xf32>
    %246 = stablehlo.convert %245 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %247 = stablehlo.slice %192 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %248 = stablehlo.slice %192 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %249 = stablehlo.broadcast_in_dim %246, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %250 = stablehlo.multiply %247, %249 : tensor<8x4096x9x64xbf16>
    %251 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %252 = stablehlo.multiply %248, %251 : tensor<8x4096x9x64xbf16>
    %253 = stablehlo.subtract %250, %252 : tensor<8x4096x9x64xbf16>
    %254 = stablehlo.broadcast_in_dim %246, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %255 = stablehlo.multiply %248, %254 : tensor<8x4096x9x64xbf16>
    %256 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %257 = stablehlo.multiply %247, %256 : tensor<8x4096x9x64xbf16>
    %258 = stablehlo.add %255, %257 : tensor<8x4096x9x64xbf16>
    %259 = stablehlo.concatenate %253, %258, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %260 = stablehlo.custom_call @Sharding(%226) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %261 = stablehlo.custom_call @Sharding(%259) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %262 = stablehlo.custom_call @Sharding(%193) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %263 = stablehlo.reduce_precision %260, format = e8m7 : tensor<8x4096x9x128xbf16>
    %264 = stablehlo.reduce_precision %261, format = e8m7 : tensor<8x4096x9x128xbf16>
    %265 = stablehlo.reduce_precision %262, format = e8m7 : tensor<8x4096x9x128xbf16>
    %266 = stablehlo.reshape %263 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %267 = stablehlo.dot_general %264, %266, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %268 = stablehlo.transpose %267, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %269 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %270 = stablehlo.broadcast_in_dim %arg203, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %271 = stablehlo.broadcast_in_dim %269, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %272 = stablehlo.broadcast_in_dim %270, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %273 = stablehlo.compare  EQ, %271, %272,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %274 = stablehlo.broadcast_in_dim %273, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %275 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %276 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_30 = stablehlo.constant dense<0> : tensor<i32>
    %277 = stablehlo.broadcast_in_dim %c_30, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %278 = stablehlo.add %275, %277 : tensor<4096x4096xi32>
    %279 = stablehlo.compare  LE, %276, %278,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %280 = stablehlo.broadcast_in_dim %279, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %281 = stablehlo.broadcast_in_dim %280, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %282 = stablehlo.and %274, %281 : tensor<8x1x1x4096x4096xi1>
    %cst_31 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_32 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %283 = call @_where(%282, %cst_31, %cst_32) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_33 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %284 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %285 = stablehlo.compare  GE, %283, %284,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_34 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %286 = call @_where_34(%285, %268, %cst_34) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> tensor<8x9x1x4096x4096xbf16>
    %287 = stablehlo.reshape %286 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_35 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %288 = stablehlo.reduce(%287 init: %cst_35) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %291 = stablehlo.subtract %287, %290 : tensor<8x9x4096x4096xbf16>
    %292 = stablehlo.exponential %291 : tensor<8x9x4096x4096xbf16>
    %293 = stablehlo.convert %292 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %294 = stablehlo.reduce(%293 init: %cst_36) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %295 = stablehlo.broadcast_in_dim %294, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %296 = stablehlo.convert %295 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %297 = stablehlo.reshape %292 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %298 = stablehlo.transpose %296, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %299 = stablehlo.dot_general %265, %297, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %300 = stablehlo.transpose %299, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %301 = stablehlo.reshape %300 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %302 = stablehlo.broadcast_in_dim %298, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %303 = stablehlo.divide %301, %302 : tensor<8x4096x9x128xbf16>
    %304 = stablehlo.custom_call @Sharding(%303) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %305 = stablehlo.dot_general %304, %arg17, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x576xbf16>
    %306 = stablehlo.reduce_precision %305, format = e8m7 : tensor<8x4096x576xbf16>
    %307 = stablehlo.custom_call @Sharding(%306) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %308 = stablehlo.add %172, %307 : tensor<8x4096x576xbf16>
    %309 = stablehlo.convert %308 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %310 = stablehlo.multiply %309, %309 : tensor<8x4096x576xf32>
    %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %311 = stablehlo.reduce(%310 init: %cst_37) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_38 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %313 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %314 = stablehlo.divide %312, %313 : tensor<8x4096x1xf32>
    %cst_39 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %315 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %316 = stablehlo.add %314, %315 : tensor<8x4096x1xf32>
    %317 = stablehlo.rsqrt %316 : tensor<8x4096x1xf32>
    %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %319 = stablehlo.multiply %309, %318 : tensor<8x4096x576xf32>
    %320 = stablehlo.convert %319 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %321 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %323 = stablehlo.multiply %320, %322 : tensor<8x4096x576xbf16>
    %324 = stablehlo.custom_call @Sharding(%323) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %325 = stablehlo.dot_general %324, %arg11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %326:2 = call @silu(%325) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %327 = stablehlo.dot_general %324, %arg12, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %328 = stablehlo.reduce_precision %327, format = e8m7 : tensor<8x4096x2016xbf16>
    %329 = stablehlo.multiply %326#0, %328 : tensor<8x4096x2016xbf16>
    %330 = stablehlo.custom_call @Sharding(%329) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %331 = stablehlo.dot_general %330, %arg13, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x576xbf16>
    %332 = stablehlo.custom_call @Sharding(%331) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %333 = stablehlo.add %332, %308 : tensor<8x4096x576xbf16>
    %334 = stablehlo.custom_call @Sharding(%333) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %335 = stablehlo.custom_call @Sharding(%334) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %336 = stablehlo.convert %335 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %337 = stablehlo.multiply %336, %336 : tensor<8x4096x576xf32>
    %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %338 = stablehlo.reduce(%337 init: %cst_40) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %339 = stablehlo.broadcast_in_dim %338, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_41 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %340 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %341 = stablehlo.divide %339, %340 : tensor<8x4096x1xf32>
    %cst_42 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %342 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %343 = stablehlo.add %341, %342 : tensor<8x4096x1xf32>
    %344 = stablehlo.rsqrt %343 : tensor<8x4096x1xf32>
    %345 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %346 = stablehlo.multiply %336, %345 : tensor<8x4096x576xf32>
    %347 = stablehlo.convert %346 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %348 = stablehlo.broadcast_in_dim %arg24, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %349 = stablehlo.broadcast_in_dim %348, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %350 = stablehlo.multiply %347, %349 : tensor<8x4096x576xbf16>
    %351 = stablehlo.custom_call @Sharding(%350) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %352 = stablehlo.custom_call @Sharding(%351) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %353 = stablehlo.custom_call @Sharding(%351) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %354 = stablehlo.dot_general %352, %arg27, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %355 = stablehlo.dot_general %353, %arg25, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %356 = stablehlo.dot_general %353, %arg28, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %357 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %358 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_43 = stablehlo.constant dense<2> : tensor<i32>
    %359 = stablehlo.broadcast_in_dim %c_43, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %360 = stablehlo.multiply %359, %358 : tensor<64xi32>
    %361 = stablehlo.convert %360 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_44 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %362 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %363 = stablehlo.divide %361, %362 : tensor<64xf32>
    %cst_45 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %364 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %365 = stablehlo.power %364, %363 : tensor<64xf32>
    %cst_46 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %366 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %367 = stablehlo.multiply %366, %365 : tensor<64xf32>
    %368 = stablehlo.convert %357 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %369 = stablehlo.broadcast_in_dim %367, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %370 = stablehlo.broadcast_in_dim %368, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %371 = stablehlo.broadcast_in_dim %369, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %372 = stablehlo.divide %370, %371 : tensor<8x4096x1x64xf32>
    %373 = stablehlo.sine %372 : tensor<8x4096x1x64xf32>
    %374 = stablehlo.convert %373 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %375 = stablehlo.cosine %372 : tensor<8x4096x1x64xf32>
    %376 = stablehlo.convert %375 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %377 = stablehlo.slice %354 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %378 = stablehlo.slice %354 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %379 = stablehlo.broadcast_in_dim %376, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %380 = stablehlo.multiply %377, %379 : tensor<8x4096x9x64xbf16>
    %381 = stablehlo.broadcast_in_dim %374, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %382 = stablehlo.multiply %378, %381 : tensor<8x4096x9x64xbf16>
    %383 = stablehlo.subtract %380, %382 : tensor<8x4096x9x64xbf16>
    %384 = stablehlo.broadcast_in_dim %376, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %385 = stablehlo.multiply %378, %384 : tensor<8x4096x9x64xbf16>
    %386 = stablehlo.broadcast_in_dim %374, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %387 = stablehlo.multiply %377, %386 : tensor<8x4096x9x64xbf16>
    %388 = stablehlo.add %385, %387 : tensor<8x4096x9x64xbf16>
    %389 = stablehlo.concatenate %383, %388, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %390 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %391 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_47 = stablehlo.constant dense<2> : tensor<i32>
    %392 = stablehlo.broadcast_in_dim %c_47, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %393 = stablehlo.multiply %392, %391 : tensor<64xi32>
    %394 = stablehlo.convert %393 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_48 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %395 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %396 = stablehlo.divide %394, %395 : tensor<64xf32>
    %cst_49 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %397 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %398 = stablehlo.power %397, %396 : tensor<64xf32>
    %cst_50 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %399 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %400 = stablehlo.multiply %399, %398 : tensor<64xf32>
    %401 = stablehlo.convert %390 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %402 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %403 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %404 = stablehlo.broadcast_in_dim %402, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %405 = stablehlo.divide %403, %404 : tensor<8x4096x1x64xf32>
    %406 = stablehlo.sine %405 : tensor<8x4096x1x64xf32>
    %407 = stablehlo.convert %406 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %408 = stablehlo.cosine %405 : tensor<8x4096x1x64xf32>
    %409 = stablehlo.convert %408 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %410 = stablehlo.slice %355 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %411 = stablehlo.slice %355 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %412 = stablehlo.broadcast_in_dim %409, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %413 = stablehlo.multiply %410, %412 : tensor<8x4096x9x64xbf16>
    %414 = stablehlo.broadcast_in_dim %407, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %415 = stablehlo.multiply %411, %414 : tensor<8x4096x9x64xbf16>
    %416 = stablehlo.subtract %413, %415 : tensor<8x4096x9x64xbf16>
    %417 = stablehlo.broadcast_in_dim %409, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %418 = stablehlo.multiply %411, %417 : tensor<8x4096x9x64xbf16>
    %419 = stablehlo.broadcast_in_dim %407, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %420 = stablehlo.multiply %410, %419 : tensor<8x4096x9x64xbf16>
    %421 = stablehlo.add %418, %420 : tensor<8x4096x9x64xbf16>
    %422 = stablehlo.concatenate %416, %421, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %423 = stablehlo.custom_call @Sharding(%389) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %424 = stablehlo.custom_call @Sharding(%422) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %425 = stablehlo.custom_call @Sharding(%356) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %426 = stablehlo.reduce_precision %423, format = e8m7 : tensor<8x4096x9x128xbf16>
    %427 = stablehlo.reduce_precision %424, format = e8m7 : tensor<8x4096x9x128xbf16>
    %428 = stablehlo.reduce_precision %425, format = e8m7 : tensor<8x4096x9x128xbf16>
    %429 = stablehlo.reshape %426 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %430 = stablehlo.dot_general %427, %429, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %431 = stablehlo.transpose %430, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %432 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %433 = stablehlo.broadcast_in_dim %arg203, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %434 = stablehlo.broadcast_in_dim %432, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %435 = stablehlo.broadcast_in_dim %433, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %436 = stablehlo.compare  EQ, %434, %435,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %437 = stablehlo.broadcast_in_dim %436, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %438 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %439 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_51 = stablehlo.constant dense<0> : tensor<i32>
    %440 = stablehlo.broadcast_in_dim %c_51, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %441 = stablehlo.add %438, %440 : tensor<4096x4096xi32>
    %442 = stablehlo.compare  LE, %439, %441,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %443 = stablehlo.broadcast_in_dim %442, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %444 = stablehlo.broadcast_in_dim %443, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %445 = stablehlo.and %437, %444 : tensor<8x1x1x4096x4096xi1>
    %cst_52 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_53 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %446 = call @_where(%445, %cst_52, %cst_53) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_54 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %447 = stablehlo.broadcast_in_dim %cst_54, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %448 = stablehlo.compare  GE, %446, %447,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_55 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %449 = call @_where_34(%448, %431, %cst_55) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> tensor<8x9x1x4096x4096xbf16>
    %450 = stablehlo.reshape %449 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_56 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %451 = stablehlo.reduce(%450 init: %cst_56) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %452 = stablehlo.broadcast_in_dim %451, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %453 = stablehlo.broadcast_in_dim %452, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %454 = stablehlo.subtract %450, %453 : tensor<8x9x4096x4096xbf16>
    %455 = stablehlo.exponential %454 : tensor<8x9x4096x4096xbf16>
    %456 = stablehlo.convert %455 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_57 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %457 = stablehlo.reduce(%456 init: %cst_57) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %459 = stablehlo.convert %458 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %460 = stablehlo.reshape %455 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %461 = stablehlo.transpose %459, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %462 = stablehlo.dot_general %428, %460, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %463 = stablehlo.transpose %462, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %464 = stablehlo.reshape %463 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %465 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %466 = stablehlo.divide %464, %465 : tensor<8x4096x9x128xbf16>
    %467 = stablehlo.custom_call @Sharding(%466) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %468 = stablehlo.dot_general %467, %arg26, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x576xbf16>
    %469 = stablehlo.reduce_precision %468, format = e8m7 : tensor<8x4096x576xbf16>
    %470 = stablehlo.custom_call @Sharding(%469) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %471 = stablehlo.add %335, %470 : tensor<8x4096x576xbf16>
    %472 = stablehlo.convert %471 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %473 = stablehlo.multiply %472, %472 : tensor<8x4096x576xf32>
    %cst_58 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %474 = stablehlo.reduce(%473 init: %cst_58) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %475 = stablehlo.broadcast_in_dim %474, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_59 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %476 = stablehlo.broadcast_in_dim %cst_59, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %477 = stablehlo.divide %475, %476 : tensor<8x4096x1xf32>
    %cst_60 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %478 = stablehlo.broadcast_in_dim %cst_60, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %479 = stablehlo.add %477, %478 : tensor<8x4096x1xf32>
    %480 = stablehlo.rsqrt %479 : tensor<8x4096x1xf32>
    %481 = stablehlo.broadcast_in_dim %480, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %482 = stablehlo.multiply %472, %481 : tensor<8x4096x576xf32>
    %483 = stablehlo.convert %482 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %484 = stablehlo.broadcast_in_dim %arg23, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %485 = stablehlo.broadcast_in_dim %484, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %486 = stablehlo.multiply %483, %485 : tensor<8x4096x576xbf16>
    %487 = stablehlo.custom_call @Sharding(%486) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %488 = stablehlo.dot_general %487, %arg20, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %489:2 = call @silu(%488) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %490 = stablehlo.dot_general %487, %arg21, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %491 = stablehlo.reduce_precision %490, format = e8m7 : tensor<8x4096x2016xbf16>
    %492 = stablehlo.multiply %489#0, %491 : tensor<8x4096x2016xbf16>
    %493 = stablehlo.custom_call @Sharding(%492) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %494 = stablehlo.dot_general %493, %arg22, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x576xbf16>
    %495 = stablehlo.custom_call @Sharding(%494) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %496 = stablehlo.add %495, %471 : tensor<8x4096x576xbf16>
    %497 = stablehlo.custom_call @Sharding(%496) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %498 = stablehlo.custom_call @Sharding(%497) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %499 = stablehlo.convert %498 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %500 = stablehlo.multiply %499, %499 : tensor<8x4096x576xf32>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %501 = stablehlo.reduce(%500 init: %cst_61) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %502 = stablehlo.broadcast_in_dim %501, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_62 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %503 = stablehlo.broadcast_in_dim %cst_62, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %504 = stablehlo.divide %502, %503 : tensor<8x4096x1xf32>
    %cst_63 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %505 = stablehlo.broadcast_in_dim %cst_63, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %506 = stablehlo.add %504, %505 : tensor<8x4096x1xf32>
    %507 = stablehlo.rsqrt %506 : tensor<8x4096x1xf32>
    %508 = stablehlo.broadcast_in_dim %507, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %509 = stablehlo.multiply %499, %508 : tensor<8x4096x576xf32>
    %510 = stablehlo.convert %509 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %511 = stablehlo.broadcast_in_dim %arg33, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %512 = stablehlo.broadcast_in_dim %511, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %513 = stablehlo.multiply %510, %512 : tensor<8x4096x576xbf16>
    %514 = stablehlo.custom_call @Sharding(%513) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %515 = stablehlo.custom_call @Sharding(%514) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %516 = stablehlo.custom_call @Sharding(%514) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %517 = stablehlo.dot_general %515, %arg36, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %518 = stablehlo.dot_general %516, %arg34, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %519 = stablehlo.dot_general %516, %arg37, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %520 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %521 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_64 = stablehlo.constant dense<2> : tensor<i32>
    %522 = stablehlo.broadcast_in_dim %c_64, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %523 = stablehlo.multiply %522, %521 : tensor<64xi32>
    %524 = stablehlo.convert %523 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_65 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %525 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %526 = stablehlo.divide %524, %525 : tensor<64xf32>
    %cst_66 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %527 = stablehlo.broadcast_in_dim %cst_66, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %528 = stablehlo.power %527, %526 : tensor<64xf32>
    %cst_67 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %529 = stablehlo.broadcast_in_dim %cst_67, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %530 = stablehlo.multiply %529, %528 : tensor<64xf32>
    %531 = stablehlo.convert %520 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %532 = stablehlo.broadcast_in_dim %530, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %533 = stablehlo.broadcast_in_dim %531, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %534 = stablehlo.broadcast_in_dim %532, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %535 = stablehlo.divide %533, %534 : tensor<8x4096x1x64xf32>
    %536 = stablehlo.sine %535 : tensor<8x4096x1x64xf32>
    %537 = stablehlo.convert %536 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %538 = stablehlo.cosine %535 : tensor<8x4096x1x64xf32>
    %539 = stablehlo.convert %538 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %540 = stablehlo.slice %517 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %541 = stablehlo.slice %517 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %542 = stablehlo.broadcast_in_dim %539, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %543 = stablehlo.multiply %540, %542 : tensor<8x4096x9x64xbf16>
    %544 = stablehlo.broadcast_in_dim %537, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %545 = stablehlo.multiply %541, %544 : tensor<8x4096x9x64xbf16>
    %546 = stablehlo.subtract %543, %545 : tensor<8x4096x9x64xbf16>
    %547 = stablehlo.broadcast_in_dim %539, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %548 = stablehlo.multiply %541, %547 : tensor<8x4096x9x64xbf16>
    %549 = stablehlo.broadcast_in_dim %537, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %550 = stablehlo.multiply %540, %549 : tensor<8x4096x9x64xbf16>
    %551 = stablehlo.add %548, %550 : tensor<8x4096x9x64xbf16>
    %552 = stablehlo.concatenate %546, %551, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %553 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %554 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_68 = stablehlo.constant dense<2> : tensor<i32>
    %555 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %556 = stablehlo.multiply %555, %554 : tensor<64xi32>
    %557 = stablehlo.convert %556 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_69 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %558 = stablehlo.broadcast_in_dim %cst_69, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %559 = stablehlo.divide %557, %558 : tensor<64xf32>
    %cst_70 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %560 = stablehlo.broadcast_in_dim %cst_70, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %561 = stablehlo.power %560, %559 : tensor<64xf32>
    %cst_71 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %562 = stablehlo.broadcast_in_dim %cst_71, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %563 = stablehlo.multiply %562, %561 : tensor<64xf32>
    %564 = stablehlo.convert %553 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %565 = stablehlo.broadcast_in_dim %563, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %566 = stablehlo.broadcast_in_dim %564, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %567 = stablehlo.broadcast_in_dim %565, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %568 = stablehlo.divide %566, %567 : tensor<8x4096x1x64xf32>
    %569 = stablehlo.sine %568 : tensor<8x4096x1x64xf32>
    %570 = stablehlo.convert %569 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %571 = stablehlo.cosine %568 : tensor<8x4096x1x64xf32>
    %572 = stablehlo.convert %571 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %573 = stablehlo.slice %518 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %574 = stablehlo.slice %518 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %575 = stablehlo.broadcast_in_dim %572, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %576 = stablehlo.multiply %573, %575 : tensor<8x4096x9x64xbf16>
    %577 = stablehlo.broadcast_in_dim %570, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %578 = stablehlo.multiply %574, %577 : tensor<8x4096x9x64xbf16>
    %579 = stablehlo.subtract %576, %578 : tensor<8x4096x9x64xbf16>
    %580 = stablehlo.broadcast_in_dim %572, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %581 = stablehlo.multiply %574, %580 : tensor<8x4096x9x64xbf16>
    %582 = stablehlo.broadcast_in_dim %570, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %583 = stablehlo.multiply %573, %582 : tensor<8x4096x9x64xbf16>
    %584 = stablehlo.add %581, %583 : tensor<8x4096x9x64xbf16>
    %585 = stablehlo.concatenate %579, %584, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %586 = stablehlo.custom_call @Sharding(%552) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %587 = stablehlo.custom_call @Sharding(%585) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %588 = stablehlo.custom_call @Sharding(%519) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %589 = stablehlo.reduce_precision %586, format = e8m7 : tensor<8x4096x9x128xbf16>
    %590 = stablehlo.reduce_precision %587, format = e8m7 : tensor<8x4096x9x128xbf16>
    %591 = stablehlo.reduce_precision %588, format = e8m7 : tensor<8x4096x9x128xbf16>
    %592 = stablehlo.reshape %589 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %593 = stablehlo.dot_general %590, %592, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %594 = stablehlo.transpose %593, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %595 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %596 = stablehlo.broadcast_in_dim %arg203, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %597 = stablehlo.broadcast_in_dim %595, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %598 = stablehlo.broadcast_in_dim %596, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %599 = stablehlo.compare  EQ, %597, %598,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %600 = stablehlo.broadcast_in_dim %599, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %601 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %602 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_72 = stablehlo.constant dense<0> : tensor<i32>
    %603 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %604 = stablehlo.add %601, %603 : tensor<4096x4096xi32>
    %605 = stablehlo.compare  LE, %602, %604,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %606 = stablehlo.broadcast_in_dim %605, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %607 = stablehlo.broadcast_in_dim %606, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %608 = stablehlo.and %600, %607 : tensor<8x1x1x4096x4096xi1>
    %cst_73 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_74 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %609 = call @_where(%608, %cst_73, %cst_74) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_75 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %610 = stablehlo.broadcast_in_dim %cst_75, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %611 = stablehlo.compare  GE, %609, %610,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_76 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %612 = call @_where_34(%611, %594, %cst_76) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> tensor<8x9x1x4096x4096xbf16>
    %613 = stablehlo.reshape %612 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_77 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %614 = stablehlo.reduce(%613 init: %cst_77) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %615 = stablehlo.broadcast_in_dim %614, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %616 = stablehlo.broadcast_in_dim %615, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %617 = stablehlo.subtract %613, %616 : tensor<8x9x4096x4096xbf16>
    %618 = stablehlo.exponential %617 : tensor<8x9x4096x4096xbf16>
    %619 = stablehlo.convert %618 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %620 = stablehlo.reduce(%619 init: %cst_78) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %621 = stablehlo.broadcast_in_dim %620, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %622 = stablehlo.convert %621 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %623 = stablehlo.reshape %618 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %624 = stablehlo.transpose %622, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %625 = stablehlo.dot_general %591, %623, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %626 = stablehlo.transpose %625, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %627 = stablehlo.reshape %626 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %628 = stablehlo.broadcast_in_dim %624, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %629 = stablehlo.divide %627, %628 : tensor<8x4096x9x128xbf16>
    %630 = stablehlo.custom_call @Sharding(%629) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %631 = stablehlo.dot_general %630, %arg35, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x576xbf16>
    %632 = stablehlo.reduce_precision %631, format = e8m7 : tensor<8x4096x576xbf16>
    %633 = stablehlo.custom_call @Sharding(%632) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %634 = stablehlo.add %498, %633 : tensor<8x4096x576xbf16>
    %635 = stablehlo.convert %634 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %636 = stablehlo.multiply %635, %635 : tensor<8x4096x576xf32>
    %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %637 = stablehlo.reduce(%636 init: %cst_79) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %638 = stablehlo.broadcast_in_dim %637, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_80 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %639 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %640 = stablehlo.divide %638, %639 : tensor<8x4096x1xf32>
    %cst_81 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %641 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %642 = stablehlo.add %640, %641 : tensor<8x4096x1xf32>
    %643 = stablehlo.rsqrt %642 : tensor<8x4096x1xf32>
    %644 = stablehlo.broadcast_in_dim %643, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %645 = stablehlo.multiply %635, %644 : tensor<8x4096x576xf32>
    %646 = stablehlo.convert %645 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %647 = stablehlo.broadcast_in_dim %arg32, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %648 = stablehlo.broadcast_in_dim %647, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %649 = stablehlo.multiply %646, %648 : tensor<8x4096x576xbf16>
    %650 = stablehlo.custom_call @Sharding(%649) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %651 = stablehlo.dot_general %650, %arg29, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %652:2 = call @silu(%651) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %653 = stablehlo.dot_general %650, %arg30, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %654 = stablehlo.reduce_precision %653, format = e8m7 : tensor<8x4096x2016xbf16>
    %655 = stablehlo.multiply %652#0, %654 : tensor<8x4096x2016xbf16>
    %656 = stablehlo.custom_call @Sharding(%655) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %657 = stablehlo.dot_general %656, %arg31, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x576xbf16>
    %658 = stablehlo.custom_call @Sharding(%657) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %659 = stablehlo.add %658, %634 : tensor<8x4096x576xbf16>
    %660 = stablehlo.custom_call @Sharding(%659) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %661 = stablehlo.custom_call @Sharding(%660) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %662 = stablehlo.convert %661 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %663 = stablehlo.multiply %662, %662 : tensor<8x4096x576xf32>
    %cst_82 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %664 = stablehlo.reduce(%663 init: %cst_82) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %665 = stablehlo.broadcast_in_dim %664, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_83 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %666 = stablehlo.broadcast_in_dim %cst_83, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %667 = stablehlo.divide %665, %666 : tensor<8x4096x1xf32>
    %cst_84 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %668 = stablehlo.broadcast_in_dim %cst_84, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %669 = stablehlo.add %667, %668 : tensor<8x4096x1xf32>
    %670 = stablehlo.rsqrt %669 : tensor<8x4096x1xf32>
    %671 = stablehlo.broadcast_in_dim %670, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %672 = stablehlo.multiply %662, %671 : tensor<8x4096x576xf32>
    %673 = stablehlo.convert %672 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %674 = stablehlo.broadcast_in_dim %arg42, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %675 = stablehlo.broadcast_in_dim %674, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %676 = stablehlo.multiply %673, %675 : tensor<8x4096x576xbf16>
    %677 = stablehlo.custom_call @Sharding(%676) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %678 = stablehlo.custom_call @Sharding(%677) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %679 = stablehlo.custom_call @Sharding(%677) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %680 = stablehlo.dot_general %678, %arg45, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %681 = stablehlo.dot_general %679, %arg43, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %682 = stablehlo.dot_general %679, %arg46, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %683 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %684 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_85 = stablehlo.constant dense<2> : tensor<i32>
    %685 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %686 = stablehlo.multiply %685, %684 : tensor<64xi32>
    %687 = stablehlo.convert %686 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_86 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %688 = stablehlo.broadcast_in_dim %cst_86, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %689 = stablehlo.divide %687, %688 : tensor<64xf32>
    %cst_87 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %690 = stablehlo.broadcast_in_dim %cst_87, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %691 = stablehlo.power %690, %689 : tensor<64xf32>
    %cst_88 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %692 = stablehlo.broadcast_in_dim %cst_88, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %693 = stablehlo.multiply %692, %691 : tensor<64xf32>
    %694 = stablehlo.convert %683 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %695 = stablehlo.broadcast_in_dim %693, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %696 = stablehlo.broadcast_in_dim %694, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %697 = stablehlo.broadcast_in_dim %695, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %698 = stablehlo.divide %696, %697 : tensor<8x4096x1x64xf32>
    %699 = stablehlo.sine %698 : tensor<8x4096x1x64xf32>
    %700 = stablehlo.convert %699 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %701 = stablehlo.cosine %698 : tensor<8x4096x1x64xf32>
    %702 = stablehlo.convert %701 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %703 = stablehlo.slice %680 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %704 = stablehlo.slice %680 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %705 = stablehlo.broadcast_in_dim %702, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %706 = stablehlo.multiply %703, %705 : tensor<8x4096x9x64xbf16>
    %707 = stablehlo.broadcast_in_dim %700, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %708 = stablehlo.multiply %704, %707 : tensor<8x4096x9x64xbf16>
    %709 = stablehlo.subtract %706, %708 : tensor<8x4096x9x64xbf16>
    %710 = stablehlo.broadcast_in_dim %702, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %711 = stablehlo.multiply %704, %710 : tensor<8x4096x9x64xbf16>
    %712 = stablehlo.broadcast_in_dim %700, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %713 = stablehlo.multiply %703, %712 : tensor<8x4096x9x64xbf16>
    %714 = stablehlo.add %711, %713 : tensor<8x4096x9x64xbf16>
    %715 = stablehlo.concatenate %709, %714, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %716 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %717 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_89 = stablehlo.constant dense<2> : tensor<i32>
    %718 = stablehlo.broadcast_in_dim %c_89, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %719 = stablehlo.multiply %718, %717 : tensor<64xi32>
    %720 = stablehlo.convert %719 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_90 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %721 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %722 = stablehlo.divide %720, %721 : tensor<64xf32>
    %cst_91 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %723 = stablehlo.broadcast_in_dim %cst_91, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %724 = stablehlo.power %723, %722 : tensor<64xf32>
    %cst_92 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %725 = stablehlo.broadcast_in_dim %cst_92, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %726 = stablehlo.multiply %725, %724 : tensor<64xf32>
    %727 = stablehlo.convert %716 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %728 = stablehlo.broadcast_in_dim %726, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %729 = stablehlo.broadcast_in_dim %727, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %730 = stablehlo.broadcast_in_dim %728, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %731 = stablehlo.divide %729, %730 : tensor<8x4096x1x64xf32>
    %732 = stablehlo.sine %731 : tensor<8x4096x1x64xf32>
    %733 = stablehlo.convert %732 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %734 = stablehlo.cosine %731 : tensor<8x4096x1x64xf32>
    %735 = stablehlo.convert %734 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %736 = stablehlo.slice %681 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %737 = stablehlo.slice %681 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %738 = stablehlo.broadcast_in_dim %735, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %739 = stablehlo.multiply %736, %738 : tensor<8x4096x9x64xbf16>
    %740 = stablehlo.broadcast_in_dim %733, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %741 = stablehlo.multiply %737, %740 : tensor<8x4096x9x64xbf16>
    %742 = stablehlo.subtract %739, %741 : tensor<8x4096x9x64xbf16>
    %743 = stablehlo.broadcast_in_dim %735, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %744 = stablehlo.multiply %737, %743 : tensor<8x4096x9x64xbf16>
    %745 = stablehlo.broadcast_in_dim %733, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %746 = stablehlo.multiply %736, %745 : tensor<8x4096x9x64xbf16>
    %747 = stablehlo.add %744, %746 : tensor<8x4096x9x64xbf16>
    %748 = stablehlo.concatenate %742, %747, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %749 = stablehlo.custom_call @Sharding(%715) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %750 = stablehlo.custom_call @Sharding(%748) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %751 = stablehlo.custom_call @Sharding(%682) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %752 = stablehlo.reduce_precision %749, format = e8m7 : tensor<8x4096x9x128xbf16>
    %753 = stablehlo.reduce_precision %750, format = e8m7 : tensor<8x4096x9x128xbf16>
    %754 = stablehlo.reduce_precision %751, format = e8m7 : tensor<8x4096x9x128xbf16>
    %755 = stablehlo.reshape %752 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %756 = stablehlo.dot_general %753, %755, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %757 = stablehlo.transpose %756, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %758 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %759 = stablehlo.broadcast_in_dim %arg203, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %760 = stablehlo.broadcast_in_dim %758, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %761 = stablehlo.broadcast_in_dim %759, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %762 = stablehlo.compare  EQ, %760, %761,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %763 = stablehlo.broadcast_in_dim %762, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %764 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %765 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_93 = stablehlo.constant dense<0> : tensor<i32>
    %766 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %767 = stablehlo.add %764, %766 : tensor<4096x4096xi32>
    %768 = stablehlo.compare  LE, %765, %767,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %769 = stablehlo.broadcast_in_dim %768, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %770 = stablehlo.broadcast_in_dim %769, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %771 = stablehlo.and %763, %770 : tensor<8x1x1x4096x4096xi1>
    %cst_94 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_95 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %772 = call @_where(%771, %cst_94, %cst_95) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_96 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %773 = stablehlo.broadcast_in_dim %cst_96, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %774 = stablehlo.compare  GE, %772, %773,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_97 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %775 = call @_where_34(%774, %757, %cst_97) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> tensor<8x9x1x4096x4096xbf16>
    %776 = stablehlo.reshape %775 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_98 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %777 = stablehlo.reduce(%776 init: %cst_98) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %778 = stablehlo.broadcast_in_dim %777, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %779 = stablehlo.broadcast_in_dim %778, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %780 = stablehlo.subtract %776, %779 : tensor<8x9x4096x4096xbf16>
    %781 = stablehlo.exponential %780 : tensor<8x9x4096x4096xbf16>
    %782 = stablehlo.convert %781 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_99 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %783 = stablehlo.reduce(%782 init: %cst_99) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %784 = stablehlo.broadcast_in_dim %783, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %785 = stablehlo.convert %784 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %786 = stablehlo.reshape %781 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %787 = stablehlo.transpose %785, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %788 = stablehlo.dot_general %754, %786, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %789 = stablehlo.transpose %788, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %790 = stablehlo.reshape %789 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %791 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %792 = stablehlo.divide %790, %791 : tensor<8x4096x9x128xbf16>
    %793 = stablehlo.custom_call @Sharding(%792) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %794 = stablehlo.dot_general %793, %arg44, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x576xbf16>
    %795 = stablehlo.reduce_precision %794, format = e8m7 : tensor<8x4096x576xbf16>
    %796 = stablehlo.custom_call @Sharding(%795) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %797 = stablehlo.add %661, %796 : tensor<8x4096x576xbf16>
    %798 = stablehlo.convert %797 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %799 = stablehlo.multiply %798, %798 : tensor<8x4096x576xf32>
    %cst_100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %800 = stablehlo.reduce(%799 init: %cst_100) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %801 = stablehlo.broadcast_in_dim %800, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_101 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %802 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %803 = stablehlo.divide %801, %802 : tensor<8x4096x1xf32>
    %cst_102 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %804 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %805 = stablehlo.add %803, %804 : tensor<8x4096x1xf32>
    %806 = stablehlo.rsqrt %805 : tensor<8x4096x1xf32>
    %807 = stablehlo.broadcast_in_dim %806, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %808 = stablehlo.multiply %798, %807 : tensor<8x4096x576xf32>
    %809 = stablehlo.convert %808 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %810 = stablehlo.broadcast_in_dim %arg41, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %811 = stablehlo.broadcast_in_dim %810, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %812 = stablehlo.multiply %809, %811 : tensor<8x4096x576xbf16>
    %813 = stablehlo.custom_call @Sharding(%812) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %814 = stablehlo.dot_general %813, %arg38, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %815:2 = call @silu(%814) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %816 = stablehlo.dot_general %813, %arg39, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %817 = stablehlo.reduce_precision %816, format = e8m7 : tensor<8x4096x2016xbf16>
    %818 = stablehlo.multiply %815#0, %817 : tensor<8x4096x2016xbf16>
    %819 = stablehlo.custom_call @Sharding(%818) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %820 = stablehlo.dot_general %819, %arg40, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x576xbf16>
    %821 = stablehlo.custom_call @Sharding(%820) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %822 = stablehlo.add %821, %797 : tensor<8x4096x576xbf16>
    %823 = stablehlo.custom_call @Sharding(%822) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %824 = stablehlo.custom_call @Sharding(%823) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %825 = stablehlo.convert %824 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %826 = stablehlo.multiply %825, %825 : tensor<8x4096x576xf32>
    %cst_103 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %827 = stablehlo.reduce(%826 init: %cst_103) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %828 = stablehlo.broadcast_in_dim %827, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_104 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %829 = stablehlo.broadcast_in_dim %cst_104, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %830 = stablehlo.divide %828, %829 : tensor<8x4096x1xf32>
    %cst_105 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %831 = stablehlo.broadcast_in_dim %cst_105, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %832 = stablehlo.add %830, %831 : tensor<8x4096x1xf32>
    %833 = stablehlo.rsqrt %832 : tensor<8x4096x1xf32>
    %834 = stablehlo.broadcast_in_dim %833, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %835 = stablehlo.multiply %825, %834 : tensor<8x4096x576xf32>
    %836 = stablehlo.convert %835 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %837 = stablehlo.broadcast_in_dim %arg51, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %838 = stablehlo.broadcast_in_dim %837, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %839 = stablehlo.multiply %836, %838 : tensor<8x4096x576xbf16>
    %840 = stablehlo.custom_call @Sharding(%839) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %841 = stablehlo.custom_call @Sharding(%840) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %842 = stablehlo.custom_call @Sharding(%840) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %843 = stablehlo.dot_general %841, %arg54, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %844 = stablehlo.dot_general %842, %arg52, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %845 = stablehlo.dot_general %842, %arg55, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %846 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %847 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_106 = stablehlo.constant dense<2> : tensor<i32>
    %848 = stablehlo.broadcast_in_dim %c_106, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %849 = stablehlo.multiply %848, %847 : tensor<64xi32>
    %850 = stablehlo.convert %849 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_107 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %851 = stablehlo.broadcast_in_dim %cst_107, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %852 = stablehlo.divide %850, %851 : tensor<64xf32>
    %cst_108 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %853 = stablehlo.broadcast_in_dim %cst_108, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %854 = stablehlo.power %853, %852 : tensor<64xf32>
    %cst_109 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %855 = stablehlo.broadcast_in_dim %cst_109, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %856 = stablehlo.multiply %855, %854 : tensor<64xf32>
    %857 = stablehlo.convert %846 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %858 = stablehlo.broadcast_in_dim %856, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %859 = stablehlo.broadcast_in_dim %857, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %860 = stablehlo.broadcast_in_dim %858, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %861 = stablehlo.divide %859, %860 : tensor<8x4096x1x64xf32>
    %862 = stablehlo.sine %861 : tensor<8x4096x1x64xf32>
    %863 = stablehlo.convert %862 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %864 = stablehlo.cosine %861 : tensor<8x4096x1x64xf32>
    %865 = stablehlo.convert %864 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %866 = stablehlo.slice %843 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %867 = stablehlo.slice %843 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %868 = stablehlo.broadcast_in_dim %865, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %869 = stablehlo.multiply %866, %868 : tensor<8x4096x9x64xbf16>
    %870 = stablehlo.broadcast_in_dim %863, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %871 = stablehlo.multiply %867, %870 : tensor<8x4096x9x64xbf16>
    %872 = stablehlo.subtract %869, %871 : tensor<8x4096x9x64xbf16>
    %873 = stablehlo.broadcast_in_dim %865, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %874 = stablehlo.multiply %867, %873 : tensor<8x4096x9x64xbf16>
    %875 = stablehlo.broadcast_in_dim %863, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %876 = stablehlo.multiply %866, %875 : tensor<8x4096x9x64xbf16>
    %877 = stablehlo.add %874, %876 : tensor<8x4096x9x64xbf16>
    %878 = stablehlo.concatenate %872, %877, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %879 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %880 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_110 = stablehlo.constant dense<2> : tensor<i32>
    %881 = stablehlo.broadcast_in_dim %c_110, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %882 = stablehlo.multiply %881, %880 : tensor<64xi32>
    %883 = stablehlo.convert %882 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_111 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %884 = stablehlo.broadcast_in_dim %cst_111, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %885 = stablehlo.divide %883, %884 : tensor<64xf32>
    %cst_112 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %886 = stablehlo.broadcast_in_dim %cst_112, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %887 = stablehlo.power %886, %885 : tensor<64xf32>
    %cst_113 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %888 = stablehlo.broadcast_in_dim %cst_113, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %889 = stablehlo.multiply %888, %887 : tensor<64xf32>
    %890 = stablehlo.convert %879 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %891 = stablehlo.broadcast_in_dim %889, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %892 = stablehlo.broadcast_in_dim %890, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %893 = stablehlo.broadcast_in_dim %891, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %894 = stablehlo.divide %892, %893 : tensor<8x4096x1x64xf32>
    %895 = stablehlo.sine %894 : tensor<8x4096x1x64xf32>
    %896 = stablehlo.convert %895 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %897 = stablehlo.cosine %894 : tensor<8x4096x1x64xf32>
    %898 = stablehlo.convert %897 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %899 = stablehlo.slice %844 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %900 = stablehlo.slice %844 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %901 = stablehlo.broadcast_in_dim %898, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %902 = stablehlo.multiply %899, %901 : tensor<8x4096x9x64xbf16>
    %903 = stablehlo.broadcast_in_dim %896, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %904 = stablehlo.multiply %900, %903 : tensor<8x4096x9x64xbf16>
    %905 = stablehlo.subtract %902, %904 : tensor<8x4096x9x64xbf16>
    %906 = stablehlo.broadcast_in_dim %898, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %907 = stablehlo.multiply %900, %906 : tensor<8x4096x9x64xbf16>
    %908 = stablehlo.broadcast_in_dim %896, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %909 = stablehlo.multiply %899, %908 : tensor<8x4096x9x64xbf16>
    %910 = stablehlo.add %907, %909 : tensor<8x4096x9x64xbf16>
    %911 = stablehlo.concatenate %905, %910, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %912 = stablehlo.custom_call @Sharding(%878) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %913 = stablehlo.custom_call @Sharding(%911) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %914 = stablehlo.custom_call @Sharding(%845) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %915 = stablehlo.reduce_precision %912, format = e8m7 : tensor<8x4096x9x128xbf16>
    %916 = stablehlo.reduce_precision %913, format = e8m7 : tensor<8x4096x9x128xbf16>
    %917 = stablehlo.reduce_precision %914, format = e8m7 : tensor<8x4096x9x128xbf16>
    %918 = stablehlo.reshape %915 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %919 = stablehlo.dot_general %916, %918, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %920 = stablehlo.transpose %919, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %921 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %922 = stablehlo.broadcast_in_dim %arg203, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %923 = stablehlo.broadcast_in_dim %921, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %924 = stablehlo.broadcast_in_dim %922, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %925 = stablehlo.compare  EQ, %923, %924,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %926 = stablehlo.broadcast_in_dim %925, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %927 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %928 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_114 = stablehlo.constant dense<0> : tensor<i32>
    %929 = stablehlo.broadcast_in_dim %c_114, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %930 = stablehlo.add %927, %929 : tensor<4096x4096xi32>
    %931 = stablehlo.compare  LE, %928, %930,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %932 = stablehlo.broadcast_in_dim %931, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %933 = stablehlo.broadcast_in_dim %932, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %934 = stablehlo.and %926, %933 : tensor<8x1x1x4096x4096xi1>
    %cst_115 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_116 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %935 = call @_where(%934, %cst_115, %cst_116) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_117 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %936 = stablehlo.broadcast_in_dim %cst_117, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %937 = stablehlo.compare  GE, %935, %936,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_118 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %938 = call @_where_34(%937, %920, %cst_118) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> tensor<8x9x1x4096x4096xbf16>
    %939 = stablehlo.reshape %938 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_119 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %940 = stablehlo.reduce(%939 init: %cst_119) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %941 = stablehlo.broadcast_in_dim %940, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %942 = stablehlo.broadcast_in_dim %941, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %943 = stablehlo.subtract %939, %942 : tensor<8x9x4096x4096xbf16>
    %944 = stablehlo.exponential %943 : tensor<8x9x4096x4096xbf16>
    %945 = stablehlo.convert %944 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %946 = stablehlo.reduce(%945 init: %cst_120) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %947 = stablehlo.broadcast_in_dim %946, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %948 = stablehlo.convert %947 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %949 = stablehlo.reshape %944 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %950 = stablehlo.transpose %948, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %951 = stablehlo.dot_general %917, %949, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %952 = stablehlo.transpose %951, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %953 = stablehlo.reshape %952 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %954 = stablehlo.broadcast_in_dim %950, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %955 = stablehlo.divide %953, %954 : tensor<8x4096x9x128xbf16>
    %956 = stablehlo.custom_call @Sharding(%955) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %957 = stablehlo.dot_general %956, %arg53, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x576xbf16>
    %958 = stablehlo.reduce_precision %957, format = e8m7 : tensor<8x4096x576xbf16>
    %959 = stablehlo.custom_call @Sharding(%958) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %960 = stablehlo.add %824, %959 : tensor<8x4096x576xbf16>
    %961 = stablehlo.convert %960 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %962 = stablehlo.multiply %961, %961 : tensor<8x4096x576xf32>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %963 = stablehlo.reduce(%962 init: %cst_121) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %964 = stablehlo.broadcast_in_dim %963, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_122 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %965 = stablehlo.broadcast_in_dim %cst_122, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %966 = stablehlo.divide %964, %965 : tensor<8x4096x1xf32>
    %cst_123 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %967 = stablehlo.broadcast_in_dim %cst_123, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %968 = stablehlo.add %966, %967 : tensor<8x4096x1xf32>
    %969 = stablehlo.rsqrt %968 : tensor<8x4096x1xf32>
    %970 = stablehlo.broadcast_in_dim %969, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %971 = stablehlo.multiply %961, %970 : tensor<8x4096x576xf32>
    %972 = stablehlo.convert %971 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %973 = stablehlo.broadcast_in_dim %arg50, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %974 = stablehlo.broadcast_in_dim %973, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %975 = stablehlo.multiply %972, %974 : tensor<8x4096x576xbf16>
    %976 = stablehlo.custom_call @Sharding(%975) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %977 = stablehlo.dot_general %976, %arg47, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %978:2 = call @silu(%977) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %979 = stablehlo.dot_general %976, %arg48, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %980 = stablehlo.reduce_precision %979, format = e8m7 : tensor<8x4096x2016xbf16>
    %981 = stablehlo.multiply %978#0, %980 : tensor<8x4096x2016xbf16>
    %982 = stablehlo.custom_call @Sharding(%981) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %983 = stablehlo.dot_general %982, %arg49, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x576xbf16>
    %984 = stablehlo.custom_call @Sharding(%983) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %985 = stablehlo.add %984, %960 : tensor<8x4096x576xbf16>
    %986 = stablehlo.custom_call @Sharding(%985) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %987 = stablehlo.custom_call @Sharding(%986) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %988 = stablehlo.convert %987 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %989 = stablehlo.multiply %988, %988 : tensor<8x4096x576xf32>
    %cst_124 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %990 = stablehlo.reduce(%989 init: %cst_124) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %991 = stablehlo.broadcast_in_dim %990, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_125 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %992 = stablehlo.broadcast_in_dim %cst_125, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %993 = stablehlo.divide %991, %992 : tensor<8x4096x1xf32>
    %cst_126 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %994 = stablehlo.broadcast_in_dim %cst_126, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %995 = stablehlo.add %993, %994 : tensor<8x4096x1xf32>
    %996 = stablehlo.rsqrt %995 : tensor<8x4096x1xf32>
    %997 = stablehlo.broadcast_in_dim %996, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %998 = stablehlo.multiply %988, %997 : tensor<8x4096x576xf32>
    %999 = stablehlo.convert %998 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1000 = stablehlo.broadcast_in_dim %arg60, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1001 = stablehlo.broadcast_in_dim %1000, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1002 = stablehlo.multiply %999, %1001 : tensor<8x4096x576xbf16>
    %1003 = stablehlo.custom_call @Sharding(%1002) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1004 = stablehlo.custom_call @Sharding(%1003) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1005 = stablehlo.custom_call @Sharding(%1003) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1006 = stablehlo.dot_general %1004, %arg63, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1007 = stablehlo.dot_general %1005, %arg61, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1008 = stablehlo.dot_general %1005, %arg64, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1009 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %1010 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_127 = stablehlo.constant dense<2> : tensor<i32>
    %1011 = stablehlo.broadcast_in_dim %c_127, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %1012 = stablehlo.multiply %1011, %1010 : tensor<64xi32>
    %1013 = stablehlo.convert %1012 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_128 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1014 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1015 = stablehlo.divide %1013, %1014 : tensor<64xf32>
    %cst_129 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %1016 = stablehlo.broadcast_in_dim %cst_129, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1017 = stablehlo.power %1016, %1015 : tensor<64xf32>
    %cst_130 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1018 = stablehlo.broadcast_in_dim %cst_130, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1019 = stablehlo.multiply %1018, %1017 : tensor<64xf32>
    %1020 = stablehlo.convert %1009 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %1021 = stablehlo.broadcast_in_dim %1019, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %1022 = stablehlo.broadcast_in_dim %1020, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %1023 = stablehlo.broadcast_in_dim %1021, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %1024 = stablehlo.divide %1022, %1023 : tensor<8x4096x1x64xf32>
    %1025 = stablehlo.sine %1024 : tensor<8x4096x1x64xf32>
    %1026 = stablehlo.convert %1025 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1027 = stablehlo.cosine %1024 : tensor<8x4096x1x64xf32>
    %1028 = stablehlo.convert %1027 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1029 = stablehlo.slice %1006 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %1030 = stablehlo.slice %1006 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %1031 = stablehlo.broadcast_in_dim %1028, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1032 = stablehlo.multiply %1029, %1031 : tensor<8x4096x9x64xbf16>
    %1033 = stablehlo.broadcast_in_dim %1026, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1034 = stablehlo.multiply %1030, %1033 : tensor<8x4096x9x64xbf16>
    %1035 = stablehlo.subtract %1032, %1034 : tensor<8x4096x9x64xbf16>
    %1036 = stablehlo.broadcast_in_dim %1028, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1037 = stablehlo.multiply %1030, %1036 : tensor<8x4096x9x64xbf16>
    %1038 = stablehlo.broadcast_in_dim %1026, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1039 = stablehlo.multiply %1029, %1038 : tensor<8x4096x9x64xbf16>
    %1040 = stablehlo.add %1037, %1039 : tensor<8x4096x9x64xbf16>
    %1041 = stablehlo.concatenate %1035, %1040, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %1042 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %1043 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_131 = stablehlo.constant dense<2> : tensor<i32>
    %1044 = stablehlo.broadcast_in_dim %c_131, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %1045 = stablehlo.multiply %1044, %1043 : tensor<64xi32>
    %1046 = stablehlo.convert %1045 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_132 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1047 = stablehlo.broadcast_in_dim %cst_132, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1048 = stablehlo.divide %1046, %1047 : tensor<64xf32>
    %cst_133 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %1049 = stablehlo.broadcast_in_dim %cst_133, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1050 = stablehlo.power %1049, %1048 : tensor<64xf32>
    %cst_134 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1051 = stablehlo.broadcast_in_dim %cst_134, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1052 = stablehlo.multiply %1051, %1050 : tensor<64xf32>
    %1053 = stablehlo.convert %1042 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %1054 = stablehlo.broadcast_in_dim %1052, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %1055 = stablehlo.broadcast_in_dim %1053, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %1056 = stablehlo.broadcast_in_dim %1054, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %1057 = stablehlo.divide %1055, %1056 : tensor<8x4096x1x64xf32>
    %1058 = stablehlo.sine %1057 : tensor<8x4096x1x64xf32>
    %1059 = stablehlo.convert %1058 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1060 = stablehlo.cosine %1057 : tensor<8x4096x1x64xf32>
    %1061 = stablehlo.convert %1060 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1062 = stablehlo.slice %1007 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %1063 = stablehlo.slice %1007 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %1064 = stablehlo.broadcast_in_dim %1061, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1065 = stablehlo.multiply %1062, %1064 : tensor<8x4096x9x64xbf16>
    %1066 = stablehlo.broadcast_in_dim %1059, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1067 = stablehlo.multiply %1063, %1066 : tensor<8x4096x9x64xbf16>
    %1068 = stablehlo.subtract %1065, %1067 : tensor<8x4096x9x64xbf16>
    %1069 = stablehlo.broadcast_in_dim %1061, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1070 = stablehlo.multiply %1063, %1069 : tensor<8x4096x9x64xbf16>
    %1071 = stablehlo.broadcast_in_dim %1059, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1072 = stablehlo.multiply %1062, %1071 : tensor<8x4096x9x64xbf16>
    %1073 = stablehlo.add %1070, %1072 : tensor<8x4096x9x64xbf16>
    %1074 = stablehlo.concatenate %1068, %1073, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %1075 = stablehlo.custom_call @Sharding(%1041) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1076 = stablehlo.custom_call @Sharding(%1074) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1077 = stablehlo.custom_call @Sharding(%1008) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1078 = stablehlo.reduce_precision %1075, format = e8m7 : tensor<8x4096x9x128xbf16>
    %1079 = stablehlo.reduce_precision %1076, format = e8m7 : tensor<8x4096x9x128xbf16>
    %1080 = stablehlo.reduce_precision %1077, format = e8m7 : tensor<8x4096x9x128xbf16>
    %1081 = stablehlo.reshape %1078 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %1082 = stablehlo.dot_general %1079, %1081, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %1083 = stablehlo.transpose %1082, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1084 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %1085 = stablehlo.broadcast_in_dim %arg203, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %1086 = stablehlo.broadcast_in_dim %1084, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %1087 = stablehlo.broadcast_in_dim %1085, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %1088 = stablehlo.compare  EQ, %1086, %1087,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %1089 = stablehlo.broadcast_in_dim %1088, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %1090 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %1091 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_135 = stablehlo.constant dense<0> : tensor<i32>
    %1092 = stablehlo.broadcast_in_dim %c_135, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %1093 = stablehlo.add %1090, %1092 : tensor<4096x4096xi32>
    %1094 = stablehlo.compare  LE, %1091, %1093,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %1095 = stablehlo.broadcast_in_dim %1094, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %1096 = stablehlo.broadcast_in_dim %1095, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %1097 = stablehlo.and %1089, %1096 : tensor<8x1x1x4096x4096xi1>
    %cst_136 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_137 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %1098 = call @_where(%1097, %cst_136, %cst_137) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_138 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %1099 = stablehlo.broadcast_in_dim %cst_138, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %1100 = stablehlo.compare  GE, %1098, %1099,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_139 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %1101 = call @_where_34(%1100, %1083, %cst_139) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> tensor<8x9x1x4096x4096xbf16>
    %1102 = stablehlo.reshape %1101 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_140 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1103 = stablehlo.reduce(%1102 init: %cst_140) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %1104 = stablehlo.broadcast_in_dim %1103, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %1105 = stablehlo.broadcast_in_dim %1104, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %1106 = stablehlo.subtract %1102, %1105 : tensor<8x9x4096x4096xbf16>
    %1107 = stablehlo.exponential %1106 : tensor<8x9x4096x4096xbf16>
    %1108 = stablehlo.convert %1107 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1109 = stablehlo.reduce(%1108 init: %cst_141) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %1110 = stablehlo.broadcast_in_dim %1109, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %1111 = stablehlo.convert %1110 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %1112 = stablehlo.reshape %1107 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1113 = stablehlo.transpose %1111, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %1114 = stablehlo.dot_general %1080, %1112, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %1115 = stablehlo.transpose %1114, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %1116 = stablehlo.reshape %1115 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1117 = stablehlo.broadcast_in_dim %1113, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %1118 = stablehlo.divide %1116, %1117 : tensor<8x4096x9x128xbf16>
    %1119 = stablehlo.custom_call @Sharding(%1118) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1120 = stablehlo.dot_general %1119, %arg62, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x576xbf16>
    %1121 = stablehlo.reduce_precision %1120, format = e8m7 : tensor<8x4096x576xbf16>
    %1122 = stablehlo.custom_call @Sharding(%1121) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1123 = stablehlo.add %987, %1122 : tensor<8x4096x576xbf16>
    %1124 = stablehlo.convert %1123 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1125 = stablehlo.multiply %1124, %1124 : tensor<8x4096x576xf32>
    %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1126 = stablehlo.reduce(%1125 init: %cst_142) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1127 = stablehlo.broadcast_in_dim %1126, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_143 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1128 = stablehlo.broadcast_in_dim %cst_143, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1129 = stablehlo.divide %1127, %1128 : tensor<8x4096x1xf32>
    %cst_144 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1130 = stablehlo.broadcast_in_dim %cst_144, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1131 = stablehlo.add %1129, %1130 : tensor<8x4096x1xf32>
    %1132 = stablehlo.rsqrt %1131 : tensor<8x4096x1xf32>
    %1133 = stablehlo.broadcast_in_dim %1132, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1134 = stablehlo.multiply %1124, %1133 : tensor<8x4096x576xf32>
    %1135 = stablehlo.convert %1134 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1136 = stablehlo.broadcast_in_dim %arg59, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1137 = stablehlo.broadcast_in_dim %1136, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1138 = stablehlo.multiply %1135, %1137 : tensor<8x4096x576xbf16>
    %1139 = stablehlo.custom_call @Sharding(%1138) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1140 = stablehlo.dot_general %1139, %arg56, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %1141:2 = call @silu(%1140) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %1142 = stablehlo.dot_general %1139, %arg57, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %1143 = stablehlo.reduce_precision %1142, format = e8m7 : tensor<8x4096x2016xbf16>
    %1144 = stablehlo.multiply %1141#0, %1143 : tensor<8x4096x2016xbf16>
    %1145 = stablehlo.custom_call @Sharding(%1144) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %1146 = stablehlo.dot_general %1145, %arg58, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x576xbf16>
    %1147 = stablehlo.custom_call @Sharding(%1146) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1148 = stablehlo.add %1147, %1123 : tensor<8x4096x576xbf16>
    %1149 = stablehlo.custom_call @Sharding(%1148) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1150 = stablehlo.convert %1149 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1151 = stablehlo.multiply %1150, %1150 : tensor<8x4096x576xf32>
    %cst_145 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1152 = stablehlo.broadcast_in_dim %cst_145, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %1153 = stablehlo.multiply %1152, %1150 : tensor<8x4096x576xf32>
    %cst_146 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1154 = stablehlo.reduce(%1151 init: %cst_146) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1155 = stablehlo.broadcast_in_dim %1154, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_147 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1156 = stablehlo.broadcast_in_dim %cst_147, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1157 = stablehlo.divide %1155, %1156 : tensor<8x4096x1xf32>
    %cst_148 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1158 = stablehlo.broadcast_in_dim %cst_148, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1159 = stablehlo.add %1157, %1158 : tensor<8x4096x1xf32>
    %1160 = stablehlo.rsqrt %1159 : tensor<8x4096x1xf32>
    %1161 = stablehlo.divide %1160, %1159 : tensor<8x4096x1xf32>
    %cst_149 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1162 = stablehlo.broadcast_in_dim %cst_149, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1163 = stablehlo.multiply %1162, %1161 : tensor<8x4096x1xf32>
    %1164 = stablehlo.broadcast_in_dim %1160, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1165 = stablehlo.multiply %1150, %1164 : tensor<8x4096x576xf32>
    %1166 = stablehlo.convert %1165 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1167 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1168 = stablehlo.broadcast_in_dim %1167, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1169 = stablehlo.multiply %1166, %1168 : tensor<8x4096x576xbf16>
    %1170 = stablehlo.dot_general %1169, %arg65, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<576x128256xbf16>) -> tensor<8x4096x128256xbf16>
    %1171 = stablehlo.custom_call @Sharding(%1170) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x128256xbf16>) -> tensor<8x4096x128256xbf16>
    %1172 = stablehlo.convert %1171 : (tensor<8x4096x128256xbf16>) -> tensor<8x4096x128256xf32>
    %1173 = call @_one_hot(%arg204) : (tensor<8x4096xi32>) -> tensor<8x4096x128256xf32>
    %cst_150 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1174 = stablehlo.reduce(%1172 init: %cst_150) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4096x128256xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1175 = stablehlo.broadcast_in_dim %1174, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %1176 = stablehlo.broadcast_in_dim %1175, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x128256xf32>
    %1177 = stablehlo.subtract %1172, %1176 : tensor<8x4096x128256xf32>
    %1178 = stablehlo.exponential %1177 : tensor<8x4096x128256xf32>
    %cst_151 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1179 = stablehlo.reduce(%1178 init: %cst_151) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x128256xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1180 = stablehlo.broadcast_in_dim %1179, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %1181 = stablehlo.log %1180 : tensor<8x4096x1xf32>
    %1182 = stablehlo.broadcast_in_dim %1181, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x128256xf32>
    %1183 = stablehlo.subtract %1177, %1182 : tensor<8x4096x128256xf32>
    %1184 = stablehlo.multiply %1173, %1183 : tensor<8x4096x128256xf32>
    %cst_152 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1185 = stablehlo.reduce(%1184 init: %cst_152) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x128256xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1186 = stablehlo.negate %1185 : tensor<8x4096xf32>
    %1187 = stablehlo.log %1180 : tensor<8x4096x1xf32>
    %1188 = stablehlo.add %1187, %1175 : tensor<8x4096x1xf32>
    %1189 = stablehlo.reshape %1188 : (tensor<8x4096x1xf32>) -> tensor<8x4096xf32>
    %1190 = stablehlo.multiply %1189, %1189 : tensor<8x4096xf32>
    %cst_153 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1191 = stablehlo.broadcast_in_dim %cst_153, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
    %1192 = stablehlo.multiply %1191, %1190 : tensor<8x4096xf32>
    %1193 = stablehlo.add %1186, %1192 : tensor<8x4096xf32>
    %1194 = stablehlo.custom_call @Sharding(%1193) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<8x4096xf32>) -> tensor<8x4096xf32>
    %c_154 = stablehlo.constant dense<0> : tensor<i32>
    %1195 = stablehlo.broadcast_in_dim %c_154, dims = [] : (tensor<i32>) -> tensor<8x4096xi32>
    %1196 = stablehlo.compare  NE, %arg205, %1195,  SIGNED : (tensor<8x4096xi32>, tensor<8x4096xi32>) -> tensor<8x4096xi1>
    %1197 = stablehlo.convert %1196 : (tensor<8x4096xi1>) -> tensor<8x4096xf32>
    %1198 = stablehlo.multiply %1194, %1197 : tensor<8x4096xf32>
    %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1199 = stablehlo.reduce(%1198 init: %cst_155) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096xf32>, tensor<f32>) -> tensor<f32>
    %c_156 = stablehlo.constant dense<0> : tensor<i32>
    %1200 = stablehlo.broadcast_in_dim %c_156, dims = [] : (tensor<i32>) -> tensor<8x4096xi32>
    %1201 = stablehlo.compare  NE, %arg205, %1200,  SIGNED : (tensor<8x4096xi32>, tensor<8x4096xi32>) -> tensor<8x4096xi1>
    %1202 = stablehlo.convert %1201 : (tensor<8x4096xi1>) -> tensor<8x4096xi32>
    %c_157 = stablehlo.constant dense<0> : tensor<i32>
    %1203 = stablehlo.reduce(%1202 init: %c_157) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096xi32>, tensor<i32>) -> tensor<i32>
    %1204 = stablehlo.convert %1203 : (tensor<i32>) -> tensor<f32>
    %cst_158 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %1205 = stablehlo.add %1204, %cst_158 : tensor<f32>
    %1206 = stablehlo.convert %1205 : tensor<f32>
    %1207 = stablehlo.divide %1199, %1206 : tensor<f32>
    %cst_159 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1208 = stablehlo.broadcast_in_dim %cst_159, dims = [] : (tensor<f32>) -> tensor<f32>
    %1209 = stablehlo.divide %1208, %1206 : tensor<f32>
    %1210 = stablehlo.broadcast_in_dim %1209, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
    %1211 = stablehlo.multiply %1210, %1197 : tensor<8x4096xf32>
    %1212 = stablehlo.custom_call @Sharding(%1211) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<8x4096xf32>) -> tensor<8x4096xf32>
    %cst_160 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1213 = stablehlo.broadcast_in_dim %cst_160, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
    %1214 = stablehlo.multiply %1213, %1189 : tensor<8x4096xf32>
    %cst_161 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1215 = stablehlo.broadcast_in_dim %cst_161, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
    %1216 = stablehlo.add %1215, %1214 : tensor<8x4096xf32>
    %1217 = stablehlo.broadcast_in_dim %1216, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %1218 = stablehlo.broadcast_in_dim %1217, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x128256xf32>
    %1219 = stablehlo.multiply %1218, %1178 : tensor<8x4096x128256xf32>
    %1220 = stablehlo.broadcast_in_dim %1180, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x128256xf32>
    %1221 = stablehlo.divide %1219, %1220 : tensor<8x4096x128256xf32>
    %1222 = stablehlo.subtract %1221, %1173 : tensor<8x4096x128256xf32>
    %1223 = stablehlo.broadcast_in_dim %1212, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %1224 = stablehlo.broadcast_in_dim %1223, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x128256xf32>
    %1225 = stablehlo.multiply %1224, %1222 : tensor<8x4096x128256xf32>
    %1226 = stablehlo.convert %1225 : (tensor<8x4096x128256xf32>) -> tensor<8x4096x128256xbf16>
    %1227 = stablehlo.custom_call @Sharding(%1226) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x128256xbf16>) -> tensor<8x4096x128256xbf16>
    %1228 = stablehlo.dot_general %1227, %1169, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x128256xbf16>, tensor<8x4096x576xbf16>) -> tensor<128256x576xbf16>
    %1229 = stablehlo.transpose %1228, dims = [1, 0] : (tensor<128256x576xbf16>) -> tensor<576x128256xbf16>
    %1230 = stablehlo.dot_general %1227, %arg65, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x128256xbf16>, tensor<576x128256xbf16>) -> tensor<8x4096x576xbf16>
    %1231 = stablehlo.multiply %1166, %1230 : tensor<8x4096x576xbf16>
    %cst_162 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1232 = stablehlo.reduce(%1231 init: %cst_162) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %1233 = stablehlo.reshape %1232 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1234 = stablehlo.broadcast_in_dim %1167, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1235 = stablehlo.multiply %1230, %1234 : tensor<8x4096x576xbf16>
    %cst_163 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1236 = stablehlo.reduce(%1233 init: %cst_163) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %1237 = stablehlo.convert %1235 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1238 = stablehlo.multiply %1150, %1237 : tensor<8x4096x576xf32>
    %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1239 = stablehlo.reduce(%1238 init: %cst_164) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1240 = stablehlo.reshape %1239 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %1241 = stablehlo.broadcast_in_dim %1160, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1242 = stablehlo.multiply %1237, %1241 : tensor<8x4096x576xf32>
    %1243 = stablehlo.multiply %1240, %1163 : tensor<8x4096x1xf32>
    %cst_165 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1244 = stablehlo.broadcast_in_dim %cst_165, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1245 = stablehlo.divide %1243, %1244 : tensor<8x4096x1xf32>
    %cst_166 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1246 = stablehlo.reduce(%1245 init: %cst_166) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1247 = stablehlo.broadcast_in_dim %1246, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %1248 = stablehlo.multiply %1247, %1153 : tensor<8x4096x576xf32>
    %1249 = stablehlo.add %1242, %1248 : tensor<8x4096x576xf32>
    %1250 = stablehlo.convert %1249 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1251:19 = stablehlo.optimization_barrier %1078, %1079, %1080, %1121, %1141#1, %1143, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %986, %arg203, %arg202, %1250 : tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096xi32>, tensor<8x4096xi32>, tensor<8x4096x576xbf16>
    %1252 = stablehlo.custom_call @Sharding(%1251#15) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1253 = stablehlo.convert %1252 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1254 = stablehlo.multiply %1253, %1253 : tensor<8x4096x576xf32>
    %cst_167 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1255 = stablehlo.broadcast_in_dim %cst_167, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %1256 = stablehlo.multiply %1255, %1253 : tensor<8x4096x576xf32>
    %cst_168 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1257 = stablehlo.reduce(%1254 init: %cst_168) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1258 = stablehlo.broadcast_in_dim %1257, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_169 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1259 = stablehlo.broadcast_in_dim %cst_169, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1260 = stablehlo.divide %1258, %1259 : tensor<8x4096x1xf32>
    %cst_170 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1261 = stablehlo.broadcast_in_dim %cst_170, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1262 = stablehlo.add %1260, %1261 : tensor<8x4096x1xf32>
    %1263 = stablehlo.rsqrt %1262 : tensor<8x4096x1xf32>
    %1264 = stablehlo.divide %1263, %1262 : tensor<8x4096x1xf32>
    %cst_171 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1265 = stablehlo.broadcast_in_dim %cst_171, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1266 = stablehlo.multiply %1265, %1264 : tensor<8x4096x1xf32>
    %1267 = stablehlo.broadcast_in_dim %1263, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1268 = stablehlo.multiply %1253, %1267 : tensor<8x4096x576xf32>
    %1269 = stablehlo.convert %1268 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1270 = stablehlo.broadcast_in_dim %1251#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1271 = stablehlo.broadcast_in_dim %1270, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1272 = stablehlo.multiply %1269, %1271 : tensor<8x4096x576xbf16>
    %1273 = stablehlo.custom_call @Sharding(%1272) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1274 = stablehlo.custom_call @Sharding(%1273) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1275 = stablehlo.custom_call @Sharding(%1273) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1276 = stablehlo.broadcast_in_dim %1251#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %1277 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_172 = stablehlo.constant dense<2> : tensor<i32>
    %1278 = stablehlo.broadcast_in_dim %c_172, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %1279 = stablehlo.multiply %1278, %1277 : tensor<64xi32>
    %1280 = stablehlo.convert %1279 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_173 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1281 = stablehlo.broadcast_in_dim %cst_173, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1282 = stablehlo.divide %1280, %1281 : tensor<64xf32>
    %cst_174 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %1283 = stablehlo.broadcast_in_dim %cst_174, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1284 = stablehlo.power %1283, %1282 : tensor<64xf32>
    %cst_175 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1285 = stablehlo.broadcast_in_dim %cst_175, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1286 = stablehlo.multiply %1285, %1284 : tensor<64xf32>
    %1287 = stablehlo.convert %1276 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %1288 = stablehlo.broadcast_in_dim %1286, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %1289 = stablehlo.broadcast_in_dim %1287, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %1290 = stablehlo.broadcast_in_dim %1288, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %1291 = stablehlo.divide %1289, %1290 : tensor<8x4096x1x64xf32>
    %1292 = stablehlo.sine %1291 : tensor<8x4096x1x64xf32>
    %1293 = stablehlo.convert %1292 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1294 = stablehlo.cosine %1291 : tensor<8x4096x1x64xf32>
    %1295 = stablehlo.convert %1294 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1296 = stablehlo.broadcast_in_dim %1251#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %1297 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_176 = stablehlo.constant dense<2> : tensor<i32>
    %1298 = stablehlo.broadcast_in_dim %c_176, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %1299 = stablehlo.multiply %1298, %1297 : tensor<64xi32>
    %1300 = stablehlo.convert %1299 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_177 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1301 = stablehlo.broadcast_in_dim %cst_177, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1302 = stablehlo.divide %1300, %1301 : tensor<64xf32>
    %cst_178 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %1303 = stablehlo.broadcast_in_dim %cst_178, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1304 = stablehlo.power %1303, %1302 : tensor<64xf32>
    %cst_179 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1305 = stablehlo.broadcast_in_dim %cst_179, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1306 = stablehlo.multiply %1305, %1304 : tensor<64xf32>
    %1307 = stablehlo.convert %1296 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %1308 = stablehlo.broadcast_in_dim %1306, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %1309 = stablehlo.broadcast_in_dim %1307, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %1310 = stablehlo.broadcast_in_dim %1308, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %1311 = stablehlo.divide %1309, %1310 : tensor<8x4096x1x64xf32>
    %1312 = stablehlo.sine %1311 : tensor<8x4096x1x64xf32>
    %1313 = stablehlo.convert %1312 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1314 = stablehlo.cosine %1311 : tensor<8x4096x1x64xf32>
    %1315 = stablehlo.convert %1314 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1316 = stablehlo.reshape %1251#0 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %1317 = stablehlo.dot_general %1251#1, %1316, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %1318 = stablehlo.transpose %1317, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1319 = stablehlo.broadcast_in_dim %1251#16, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %1320 = stablehlo.broadcast_in_dim %1251#16, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %1321 = stablehlo.broadcast_in_dim %1319, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %1322 = stablehlo.broadcast_in_dim %1320, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %1323 = stablehlo.compare  EQ, %1321, %1322,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %1324 = stablehlo.broadcast_in_dim %1323, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %1325 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %1326 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_180 = stablehlo.constant dense<0> : tensor<i32>
    %1327 = stablehlo.broadcast_in_dim %c_180, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %1328 = stablehlo.add %1325, %1327 : tensor<4096x4096xi32>
    %1329 = stablehlo.compare  LE, %1326, %1328,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %1330 = stablehlo.broadcast_in_dim %1329, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %1331 = stablehlo.broadcast_in_dim %1330, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %1332 = stablehlo.and %1324, %1331 : tensor<8x1x1x4096x4096xi1>
    %cst_181 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_182 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %1333 = call @_where_115(%1332, %cst_181, %cst_182) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_183 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %1334 = stablehlo.broadcast_in_dim %cst_183, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %1335 = stablehlo.compare  GE, %1333, %1334,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_184 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %1336:2 = call @_where_116(%1335, %1318, %cst_184) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> (tensor<8x9x1x4096x4096xbf16>, tensor<8x9x1x4096x4096xi1>)
    %1337 = stablehlo.reshape %1336#0 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_185 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1338 = stablehlo.reduce(%1337 init: %cst_185) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %1339 = stablehlo.reshape %1338 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %1340 = stablehlo.broadcast_in_dim %1339, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %1341 = stablehlo.compare  EQ, %1337, %1340,  FLOAT : (tensor<8x9x4096x4096xbf16>, tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xi1>
    %1342 = stablehlo.convert %1341 : (tensor<8x9x4096x4096xi1>) -> tensor<8x9x4096x4096xbf16>
    %cst_186 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1343 = stablehlo.reduce(%1342 init: %cst_186) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %1344 = stablehlo.broadcast_in_dim %1338, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %1345 = stablehlo.broadcast_in_dim %1344, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %1346 = stablehlo.subtract %1337, %1345 : tensor<8x9x4096x4096xbf16>
    %1347 = stablehlo.exponential %1346 : tensor<8x9x4096x4096xbf16>
    %1348 = stablehlo.convert %1347 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_187 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1349 = stablehlo.reduce(%1348 init: %cst_187) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %1350 = stablehlo.broadcast_in_dim %1349, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %1351 = stablehlo.convert %1350 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %1352 = stablehlo.reshape %1347 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1353 = stablehlo.transpose %1351, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %1354 = stablehlo.dot_general %1251#2, %1352, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %1355 = stablehlo.transpose %1354, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %1356 = stablehlo.reshape %1355 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1357 = stablehlo.broadcast_in_dim %1353, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %1358 = stablehlo.divide %1356, %1357 : tensor<8x4096x9x128xbf16>
    %1359 = stablehlo.multiply %1353, %1353 : tensor<8x4096x9x1xbf16>
    %cst_188 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1360 = stablehlo.broadcast_in_dim %cst_188, dims = [] : (tensor<bf16>) -> tensor<8x4096x9x1xbf16>
    %1361 = stablehlo.divide %1360, %1359 : tensor<8x4096x9x1xbf16>
    %1362 = stablehlo.custom_call @Sharding(%1358) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1363 = stablehlo.custom_call @Sharding(%1251#3) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1364 = stablehlo.add %1252, %1363 : tensor<8x4096x576xbf16>
    %1365 = stablehlo.convert %1364 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1366 = stablehlo.multiply %1365, %1365 : tensor<8x4096x576xf32>
    %cst_189 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1367 = stablehlo.broadcast_in_dim %cst_189, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %1368 = stablehlo.multiply %1367, %1365 : tensor<8x4096x576xf32>
    %cst_190 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1369 = stablehlo.reduce(%1366 init: %cst_190) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1370 = stablehlo.broadcast_in_dim %1369, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_191 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1371 = stablehlo.broadcast_in_dim %cst_191, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1372 = stablehlo.divide %1370, %1371 : tensor<8x4096x1xf32>
    %cst_192 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1373 = stablehlo.broadcast_in_dim %cst_192, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1374 = stablehlo.add %1372, %1373 : tensor<8x4096x1xf32>
    %1375 = stablehlo.rsqrt %1374 : tensor<8x4096x1xf32>
    %1376 = stablehlo.divide %1375, %1374 : tensor<8x4096x1xf32>
    %cst_193 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1377 = stablehlo.broadcast_in_dim %cst_193, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1378 = stablehlo.multiply %1377, %1376 : tensor<8x4096x1xf32>
    %1379 = stablehlo.broadcast_in_dim %1375, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1380 = stablehlo.multiply %1365, %1379 : tensor<8x4096x576xf32>
    %1381 = stablehlo.convert %1380 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1382 = stablehlo.broadcast_in_dim %1251#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1383 = stablehlo.broadcast_in_dim %1382, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1384 = stablehlo.multiply %1381, %1383 : tensor<8x4096x576xbf16>
    %1385 = stablehlo.custom_call @Sharding(%1384) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1386:3 = call @silu_124(%1251#4) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %1387 = stablehlo.multiply %1386#0, %1251#5 : tensor<8x4096x2016xbf16>
    %1388 = stablehlo.custom_call @Sharding(%1387) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %1389 = stablehlo.custom_call @Sharding(%1251#18) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1390 = stablehlo.custom_call @Sharding(%1389) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1391 = stablehlo.dot_general %1390, %1388, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %1392 = stablehlo.transpose %1391, dims = [1, 0] : (tensor<576x2016xbf16>) -> tensor<2016x576xbf16>
    %1393 = stablehlo.dot_general %1390, %1251#8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x2016xbf16>
    %1394 = stablehlo.custom_call @Sharding(%1393) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %1395 = stablehlo.multiply %1386#0, %1394 : tensor<8x4096x2016xbf16>
    %1396 = stablehlo.multiply %1394, %1251#5 : tensor<8x4096x2016xbf16>
    %1397 = stablehlo.dot_general %1395, %1385, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %1398 = stablehlo.transpose %1397, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %1399 = stablehlo.dot_general %1395, %1251#7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %1400 = call @silu_133(%1386#1, %1251#4, %1386#2, %1396) : (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %1401 = stablehlo.dot_general %1400, %1385, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %1402 = stablehlo.transpose %1401, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %1403 = stablehlo.dot_general %1400, %1251#6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %1404 = stablehlo.add %1399, %1403 : tensor<8x4096x576xbf16>
    %1405 = stablehlo.custom_call @Sharding(%1404) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1406 = stablehlo.multiply %1381, %1405 : tensor<8x4096x576xbf16>
    %cst_194 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1407 = stablehlo.reduce(%1406 init: %cst_194) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %1408 = stablehlo.reshape %1407 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1409 = stablehlo.broadcast_in_dim %1382, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1410 = stablehlo.multiply %1405, %1409 : tensor<8x4096x576xbf16>
    %cst_195 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1411 = stablehlo.reduce(%1408 init: %cst_195) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %1412 = stablehlo.convert %1410 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1413 = stablehlo.multiply %1365, %1412 : tensor<8x4096x576xf32>
    %cst_196 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1414 = stablehlo.reduce(%1413 init: %cst_196) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1415 = stablehlo.reshape %1414 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %1416 = stablehlo.broadcast_in_dim %1375, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1417 = stablehlo.multiply %1412, %1416 : tensor<8x4096x576xf32>
    %1418 = stablehlo.multiply %1415, %1378 : tensor<8x4096x1xf32>
    %cst_197 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1419 = stablehlo.broadcast_in_dim %cst_197, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1420 = stablehlo.divide %1418, %1419 : tensor<8x4096x1xf32>
    %cst_198 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1421 = stablehlo.reduce(%1420 init: %cst_198) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1422 = stablehlo.broadcast_in_dim %1421, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %1423 = stablehlo.multiply %1422, %1368 : tensor<8x4096x576xf32>
    %1424 = stablehlo.add %1417, %1423 : tensor<8x4096x576xf32>
    %1425 = stablehlo.convert %1424 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1426 = stablehlo.add %1389, %1425 : tensor<8x4096x576xbf16>
    %1427 = stablehlo.custom_call @Sharding(%1426) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1428 = stablehlo.dot_general %1427, %1362, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1429 = stablehlo.transpose %1428, dims = [1, 2, 0] : (tensor<576x9x128xbf16>) -> tensor<9x128x576xbf16>
    %1430 = stablehlo.dot_general %1427, %1251#12, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x9x128xbf16>
    %1431 = stablehlo.custom_call @Sharding(%1430) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1432 = stablehlo.broadcast_in_dim %1361, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %1433 = stablehlo.multiply %1431, %1432 : tensor<8x4096x9x128xbf16>
    %1434 = stablehlo.multiply %1433, %1356 : tensor<8x4096x9x128xbf16>
    %cst_199 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1435 = stablehlo.reduce(%1434 init: %cst_199) applies stablehlo.add across dimensions = [3] : (tensor<8x4096x9x128xbf16>, tensor<bf16>) -> tensor<8x4096x9xbf16>
    %1436 = stablehlo.reshape %1435 : (tensor<8x4096x9xbf16>) -> tensor<8x4096x9x1xbf16>
    %1437 = stablehlo.negate %1436 : tensor<8x4096x9x1xbf16>
    %1438 = stablehlo.broadcast_in_dim %1353, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %1439 = stablehlo.divide %1431, %1438 : tensor<8x4096x9x128xbf16>
    %1440 = stablehlo.reshape %1439 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %1441 = stablehlo.transpose %1440, dims = [0, 2, 4, 3, 1] : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %1442 = stablehlo.dot_general %1441, %1251#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1443 = stablehlo.dot_general %1441, %1352, batching_dims = [0, 1] x [0, 1], contracting_dims = [3, 4] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x4096xbf16>
    %1444 = stablehlo.transpose %1443, dims = [0, 3, 1, 2] : (tensor<8x9x128x4096xbf16>) -> tensor<8x4096x9x128xbf16>
    %1445 = stablehlo.transpose %1437, dims = [0, 2, 1, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x9x4096x1xbf16>
    %1446 = stablehlo.reshape %1442 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %1447 = stablehlo.convert %1445 : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x1xf32>
    %cst_200 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1448 = stablehlo.reduce(%1447 init: %cst_200) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %1449 = stablehlo.broadcast_in_dim %1448, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x4096xf32>
    %1450 = stablehlo.convert %1449 : (tensor<8x9x4096x4096xf32>) -> tensor<8x9x4096x4096xbf16>
    %1451 = stablehlo.add %1446, %1450 : tensor<8x9x4096x4096xbf16>
    %1452 = stablehlo.multiply %1451, %1347 : tensor<8x9x4096x4096xbf16>
    %1453 = stablehlo.negate %1452 : tensor<8x9x4096x4096xbf16>
    %cst_201 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1454 = stablehlo.reduce(%1453 init: %cst_201) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %1455 = stablehlo.reshape %1454 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %cst_202 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1456 = stablehlo.reduce(%1455 init: %cst_202) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %1457 = stablehlo.divide %1456, %1343 : tensor<8x9x4096xbf16>
    %1458 = stablehlo.broadcast_in_dim %1457, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %1459 = stablehlo.multiply %1458, %1342 : tensor<8x9x4096x4096xbf16>
    %1460 = stablehlo.add %1452, %1459 : tensor<8x9x4096x4096xbf16>
    %1461 = stablehlo.reshape %1460 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1462 = call @_where_163(%1336#1, %1461) : (tensor<8x9x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1463 = stablehlo.transpose %1462, dims = [0, 1, 4, 3, 2] : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %1464 = stablehlo.dot_general %1463, %1251#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x4096x1x128xbf16>
    %1465 = stablehlo.transpose %1464, dims = [0, 2, 1, 3, 4] : (tensor<8x9x4096x1x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %1466 = stablehlo.dot_general %1463, %1316, batching_dims = [0, 1] x [0, 2], contracting_dims = [3, 4] x [1, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x128xbf16>
    %1467 = stablehlo.transpose %1466, dims = [0, 2, 1, 3] : (tensor<8x9x4096x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1468 = stablehlo.reshape %1465 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1469 = stablehlo.custom_call @Sharding(%1444) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1470 = stablehlo.custom_call @Sharding(%1467) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1471 = stablehlo.custom_call @Sharding(%1468) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1472 = stablehlo.slice %1470 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %1473 = stablehlo.slice %1470 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %1474 = stablehlo.broadcast_in_dim %1313, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1475 = stablehlo.multiply %1473, %1474 : tensor<8x4096x9x64xbf16>
    %1476 = stablehlo.broadcast_in_dim %1315, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1477 = stablehlo.multiply %1473, %1476 : tensor<8x4096x9x64xbf16>
    %1478 = stablehlo.negate %1472 : tensor<8x4096x9x64xbf16>
    %1479 = stablehlo.broadcast_in_dim %1313, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1480 = stablehlo.multiply %1478, %1479 : tensor<8x4096x9x64xbf16>
    %1481 = stablehlo.add %1477, %1480 : tensor<8x4096x9x64xbf16>
    %1482 = stablehlo.broadcast_in_dim %1315, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1483 = stablehlo.multiply %1472, %1482 : tensor<8x4096x9x64xbf16>
    %1484 = stablehlo.add %1475, %1483 : tensor<8x4096x9x64xbf16>
    %1485 = stablehlo.concatenate %1484, %1481, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %1486 = stablehlo.slice %1471 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %1487 = stablehlo.slice %1471 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %1488 = stablehlo.broadcast_in_dim %1293, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1489 = stablehlo.multiply %1487, %1488 : tensor<8x4096x9x64xbf16>
    %1490 = stablehlo.broadcast_in_dim %1295, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1491 = stablehlo.multiply %1487, %1490 : tensor<8x4096x9x64xbf16>
    %1492 = stablehlo.negate %1486 : tensor<8x4096x9x64xbf16>
    %1493 = stablehlo.broadcast_in_dim %1293, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1494 = stablehlo.multiply %1492, %1493 : tensor<8x4096x9x64xbf16>
    %1495 = stablehlo.add %1491, %1494 : tensor<8x4096x9x64xbf16>
    %1496 = stablehlo.broadcast_in_dim %1295, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1497 = stablehlo.multiply %1486, %1496 : tensor<8x4096x9x64xbf16>
    %1498 = stablehlo.add %1489, %1497 : tensor<8x4096x9x64xbf16>
    %1499 = stablehlo.concatenate %1498, %1495, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %1500 = stablehlo.dot_general %1469, %1275, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %1501 = stablehlo.transpose %1500, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %1502 = stablehlo.dot_general %1469, %1251#14, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %1503 = stablehlo.dot_general %1485, %1275, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %1504 = stablehlo.transpose %1503, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %1505 = stablehlo.dot_general %1485, %1251#11, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %1506 = stablehlo.add %1502, %1505 : tensor<8x4096x576xbf16>
    %1507 = stablehlo.dot_general %1499, %1274, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %1508 = stablehlo.transpose %1507, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %1509 = stablehlo.dot_general %1499, %1251#13, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %1510 = stablehlo.custom_call @Sharding(%1506) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1511 = stablehlo.custom_call @Sharding(%1509) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1512 = stablehlo.add %1510, %1511 : tensor<8x4096x576xbf16>
    %1513 = stablehlo.custom_call @Sharding(%1512) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1514 = stablehlo.multiply %1269, %1513 : tensor<8x4096x576xbf16>
    %cst_203 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1515 = stablehlo.reduce(%1514 init: %cst_203) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %1516 = stablehlo.reshape %1515 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1517 = stablehlo.broadcast_in_dim %1270, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1518 = stablehlo.multiply %1513, %1517 : tensor<8x4096x576xbf16>
    %cst_204 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1519 = stablehlo.reduce(%1516 init: %cst_204) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %1520 = stablehlo.convert %1518 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1521 = stablehlo.multiply %1253, %1520 : tensor<8x4096x576xf32>
    %cst_205 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1522 = stablehlo.reduce(%1521 init: %cst_205) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1523 = stablehlo.reshape %1522 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %1524 = stablehlo.broadcast_in_dim %1263, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1525 = stablehlo.multiply %1520, %1524 : tensor<8x4096x576xf32>
    %1526 = stablehlo.multiply %1523, %1266 : tensor<8x4096x1xf32>
    %cst_206 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1527 = stablehlo.broadcast_in_dim %cst_206, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1528 = stablehlo.divide %1526, %1527 : tensor<8x4096x1xf32>
    %cst_207 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1529 = stablehlo.reduce(%1528 init: %cst_207) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1530 = stablehlo.broadcast_in_dim %1529, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %1531 = stablehlo.multiply %1530, %1256 : tensor<8x4096x576xf32>
    %1532 = stablehlo.add %1525, %1531 : tensor<8x4096x576xf32>
    %1533 = stablehlo.convert %1532 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1534 = stablehlo.add %1426, %1533 : tensor<8x4096x576xbf16>
    %1535 = stablehlo.custom_call @Sharding(%1534) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1536:19 = stablehlo.optimization_barrier %915, %916, %917, %958, %978#1, %980, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %823, %arg203, %arg202, %1535 : tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096xi32>, tensor<8x4096xi32>, tensor<8x4096x576xbf16>
    %1537 = stablehlo.custom_call @Sharding(%1536#15) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1538 = stablehlo.convert %1537 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1539 = stablehlo.multiply %1538, %1538 : tensor<8x4096x576xf32>
    %cst_208 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1540 = stablehlo.broadcast_in_dim %cst_208, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %1541 = stablehlo.multiply %1540, %1538 : tensor<8x4096x576xf32>
    %cst_209 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1542 = stablehlo.reduce(%1539 init: %cst_209) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1543 = stablehlo.broadcast_in_dim %1542, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_210 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1544 = stablehlo.broadcast_in_dim %cst_210, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1545 = stablehlo.divide %1543, %1544 : tensor<8x4096x1xf32>
    %cst_211 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1546 = stablehlo.broadcast_in_dim %cst_211, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1547 = stablehlo.add %1545, %1546 : tensor<8x4096x1xf32>
    %1548 = stablehlo.rsqrt %1547 : tensor<8x4096x1xf32>
    %1549 = stablehlo.divide %1548, %1547 : tensor<8x4096x1xf32>
    %cst_212 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1550 = stablehlo.broadcast_in_dim %cst_212, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1551 = stablehlo.multiply %1550, %1549 : tensor<8x4096x1xf32>
    %1552 = stablehlo.broadcast_in_dim %1548, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1553 = stablehlo.multiply %1538, %1552 : tensor<8x4096x576xf32>
    %1554 = stablehlo.convert %1553 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1555 = stablehlo.broadcast_in_dim %1536#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1556 = stablehlo.broadcast_in_dim %1555, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1557 = stablehlo.multiply %1554, %1556 : tensor<8x4096x576xbf16>
    %1558 = stablehlo.custom_call @Sharding(%1557) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1559 = stablehlo.custom_call @Sharding(%1558) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1560 = stablehlo.custom_call @Sharding(%1558) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1561 = stablehlo.broadcast_in_dim %1536#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %1562 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_213 = stablehlo.constant dense<2> : tensor<i32>
    %1563 = stablehlo.broadcast_in_dim %c_213, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %1564 = stablehlo.multiply %1563, %1562 : tensor<64xi32>
    %1565 = stablehlo.convert %1564 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_214 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1566 = stablehlo.broadcast_in_dim %cst_214, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1567 = stablehlo.divide %1565, %1566 : tensor<64xf32>
    %cst_215 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %1568 = stablehlo.broadcast_in_dim %cst_215, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1569 = stablehlo.power %1568, %1567 : tensor<64xf32>
    %cst_216 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1570 = stablehlo.broadcast_in_dim %cst_216, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1571 = stablehlo.multiply %1570, %1569 : tensor<64xf32>
    %1572 = stablehlo.convert %1561 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %1573 = stablehlo.broadcast_in_dim %1571, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %1574 = stablehlo.broadcast_in_dim %1572, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %1575 = stablehlo.broadcast_in_dim %1573, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %1576 = stablehlo.divide %1574, %1575 : tensor<8x4096x1x64xf32>
    %1577 = stablehlo.sine %1576 : tensor<8x4096x1x64xf32>
    %1578 = stablehlo.convert %1577 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1579 = stablehlo.cosine %1576 : tensor<8x4096x1x64xf32>
    %1580 = stablehlo.convert %1579 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1581 = stablehlo.broadcast_in_dim %1536#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %1582 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_217 = stablehlo.constant dense<2> : tensor<i32>
    %1583 = stablehlo.broadcast_in_dim %c_217, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %1584 = stablehlo.multiply %1583, %1582 : tensor<64xi32>
    %1585 = stablehlo.convert %1584 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_218 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1586 = stablehlo.broadcast_in_dim %cst_218, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1587 = stablehlo.divide %1585, %1586 : tensor<64xf32>
    %cst_219 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %1588 = stablehlo.broadcast_in_dim %cst_219, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1589 = stablehlo.power %1588, %1587 : tensor<64xf32>
    %cst_220 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1590 = stablehlo.broadcast_in_dim %cst_220, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1591 = stablehlo.multiply %1590, %1589 : tensor<64xf32>
    %1592 = stablehlo.convert %1581 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %1593 = stablehlo.broadcast_in_dim %1591, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %1594 = stablehlo.broadcast_in_dim %1592, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %1595 = stablehlo.broadcast_in_dim %1593, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %1596 = stablehlo.divide %1594, %1595 : tensor<8x4096x1x64xf32>
    %1597 = stablehlo.sine %1596 : tensor<8x4096x1x64xf32>
    %1598 = stablehlo.convert %1597 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1599 = stablehlo.cosine %1596 : tensor<8x4096x1x64xf32>
    %1600 = stablehlo.convert %1599 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1601 = stablehlo.reshape %1536#0 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %1602 = stablehlo.dot_general %1536#1, %1601, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %1603 = stablehlo.transpose %1602, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1604 = stablehlo.broadcast_in_dim %1536#16, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %1605 = stablehlo.broadcast_in_dim %1536#16, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %1606 = stablehlo.broadcast_in_dim %1604, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %1607 = stablehlo.broadcast_in_dim %1605, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %1608 = stablehlo.compare  EQ, %1606, %1607,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %1609 = stablehlo.broadcast_in_dim %1608, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %1610 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %1611 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_221 = stablehlo.constant dense<0> : tensor<i32>
    %1612 = stablehlo.broadcast_in_dim %c_221, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %1613 = stablehlo.add %1610, %1612 : tensor<4096x4096xi32>
    %1614 = stablehlo.compare  LE, %1611, %1613,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %1615 = stablehlo.broadcast_in_dim %1614, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %1616 = stablehlo.broadcast_in_dim %1615, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %1617 = stablehlo.and %1609, %1616 : tensor<8x1x1x4096x4096xi1>
    %cst_222 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_223 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %1618 = call @_where_115(%1617, %cst_222, %cst_223) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_224 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %1619 = stablehlo.broadcast_in_dim %cst_224, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %1620 = stablehlo.compare  GE, %1618, %1619,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_225 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %1621:2 = call @_where_116(%1620, %1603, %cst_225) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> (tensor<8x9x1x4096x4096xbf16>, tensor<8x9x1x4096x4096xi1>)
    %1622 = stablehlo.reshape %1621#0 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_226 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1623 = stablehlo.reduce(%1622 init: %cst_226) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %1624 = stablehlo.reshape %1623 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %1625 = stablehlo.broadcast_in_dim %1624, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %1626 = stablehlo.compare  EQ, %1622, %1625,  FLOAT : (tensor<8x9x4096x4096xbf16>, tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xi1>
    %1627 = stablehlo.convert %1626 : (tensor<8x9x4096x4096xi1>) -> tensor<8x9x4096x4096xbf16>
    %cst_227 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1628 = stablehlo.reduce(%1627 init: %cst_227) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %1629 = stablehlo.broadcast_in_dim %1623, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %1630 = stablehlo.broadcast_in_dim %1629, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %1631 = stablehlo.subtract %1622, %1630 : tensor<8x9x4096x4096xbf16>
    %1632 = stablehlo.exponential %1631 : tensor<8x9x4096x4096xbf16>
    %1633 = stablehlo.convert %1632 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_228 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1634 = stablehlo.reduce(%1633 init: %cst_228) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %1635 = stablehlo.broadcast_in_dim %1634, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %1636 = stablehlo.convert %1635 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %1637 = stablehlo.reshape %1632 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1638 = stablehlo.transpose %1636, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %1639 = stablehlo.dot_general %1536#2, %1637, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %1640 = stablehlo.transpose %1639, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %1641 = stablehlo.reshape %1640 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1642 = stablehlo.broadcast_in_dim %1638, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %1643 = stablehlo.divide %1641, %1642 : tensor<8x4096x9x128xbf16>
    %1644 = stablehlo.multiply %1638, %1638 : tensor<8x4096x9x1xbf16>
    %cst_229 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1645 = stablehlo.broadcast_in_dim %cst_229, dims = [] : (tensor<bf16>) -> tensor<8x4096x9x1xbf16>
    %1646 = stablehlo.divide %1645, %1644 : tensor<8x4096x9x1xbf16>
    %1647 = stablehlo.custom_call @Sharding(%1643) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1648 = stablehlo.custom_call @Sharding(%1536#3) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1649 = stablehlo.add %1537, %1648 : tensor<8x4096x576xbf16>
    %1650 = stablehlo.convert %1649 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1651 = stablehlo.multiply %1650, %1650 : tensor<8x4096x576xf32>
    %cst_230 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1652 = stablehlo.broadcast_in_dim %cst_230, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %1653 = stablehlo.multiply %1652, %1650 : tensor<8x4096x576xf32>
    %cst_231 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1654 = stablehlo.reduce(%1651 init: %cst_231) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1655 = stablehlo.broadcast_in_dim %1654, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_232 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1656 = stablehlo.broadcast_in_dim %cst_232, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1657 = stablehlo.divide %1655, %1656 : tensor<8x4096x1xf32>
    %cst_233 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1658 = stablehlo.broadcast_in_dim %cst_233, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1659 = stablehlo.add %1657, %1658 : tensor<8x4096x1xf32>
    %1660 = stablehlo.rsqrt %1659 : tensor<8x4096x1xf32>
    %1661 = stablehlo.divide %1660, %1659 : tensor<8x4096x1xf32>
    %cst_234 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1662 = stablehlo.broadcast_in_dim %cst_234, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1663 = stablehlo.multiply %1662, %1661 : tensor<8x4096x1xf32>
    %1664 = stablehlo.broadcast_in_dim %1660, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1665 = stablehlo.multiply %1650, %1664 : tensor<8x4096x576xf32>
    %1666 = stablehlo.convert %1665 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1667 = stablehlo.broadcast_in_dim %1536#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1668 = stablehlo.broadcast_in_dim %1667, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1669 = stablehlo.multiply %1666, %1668 : tensor<8x4096x576xbf16>
    %1670 = stablehlo.custom_call @Sharding(%1669) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1671:3 = call @silu_124(%1536#4) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %1672 = stablehlo.multiply %1671#0, %1536#5 : tensor<8x4096x2016xbf16>
    %1673 = stablehlo.custom_call @Sharding(%1672) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %1674 = stablehlo.custom_call @Sharding(%1536#18) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1675 = stablehlo.custom_call @Sharding(%1674) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1676 = stablehlo.dot_general %1675, %1673, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %1677 = stablehlo.transpose %1676, dims = [1, 0] : (tensor<576x2016xbf16>) -> tensor<2016x576xbf16>
    %1678 = stablehlo.dot_general %1675, %1536#8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x2016xbf16>
    %1679 = stablehlo.custom_call @Sharding(%1678) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %1680 = stablehlo.multiply %1671#0, %1679 : tensor<8x4096x2016xbf16>
    %1681 = stablehlo.multiply %1679, %1536#5 : tensor<8x4096x2016xbf16>
    %1682 = stablehlo.dot_general %1680, %1670, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %1683 = stablehlo.transpose %1682, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %1684 = stablehlo.dot_general %1680, %1536#7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %1685 = call @silu_133(%1671#1, %1536#4, %1671#2, %1681) : (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %1686 = stablehlo.dot_general %1685, %1670, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %1687 = stablehlo.transpose %1686, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %1688 = stablehlo.dot_general %1685, %1536#6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %1689 = stablehlo.add %1684, %1688 : tensor<8x4096x576xbf16>
    %1690 = stablehlo.custom_call @Sharding(%1689) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1691 = stablehlo.multiply %1666, %1690 : tensor<8x4096x576xbf16>
    %cst_235 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1692 = stablehlo.reduce(%1691 init: %cst_235) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %1693 = stablehlo.reshape %1692 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1694 = stablehlo.broadcast_in_dim %1667, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1695 = stablehlo.multiply %1690, %1694 : tensor<8x4096x576xbf16>
    %cst_236 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1696 = stablehlo.reduce(%1693 init: %cst_236) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %1697 = stablehlo.convert %1695 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1698 = stablehlo.multiply %1650, %1697 : tensor<8x4096x576xf32>
    %cst_237 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1699 = stablehlo.reduce(%1698 init: %cst_237) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1700 = stablehlo.reshape %1699 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %1701 = stablehlo.broadcast_in_dim %1660, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1702 = stablehlo.multiply %1697, %1701 : tensor<8x4096x576xf32>
    %1703 = stablehlo.multiply %1700, %1663 : tensor<8x4096x1xf32>
    %cst_238 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1704 = stablehlo.broadcast_in_dim %cst_238, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1705 = stablehlo.divide %1703, %1704 : tensor<8x4096x1xf32>
    %cst_239 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1706 = stablehlo.reduce(%1705 init: %cst_239) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1707 = stablehlo.broadcast_in_dim %1706, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %1708 = stablehlo.multiply %1707, %1653 : tensor<8x4096x576xf32>
    %1709 = stablehlo.add %1702, %1708 : tensor<8x4096x576xf32>
    %1710 = stablehlo.convert %1709 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1711 = stablehlo.add %1674, %1710 : tensor<8x4096x576xbf16>
    %1712 = stablehlo.custom_call @Sharding(%1711) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1713 = stablehlo.dot_general %1712, %1647, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1714 = stablehlo.transpose %1713, dims = [1, 2, 0] : (tensor<576x9x128xbf16>) -> tensor<9x128x576xbf16>
    %1715 = stablehlo.dot_general %1712, %1536#12, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x9x128xbf16>
    %1716 = stablehlo.custom_call @Sharding(%1715) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1717 = stablehlo.broadcast_in_dim %1646, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %1718 = stablehlo.multiply %1716, %1717 : tensor<8x4096x9x128xbf16>
    %1719 = stablehlo.multiply %1718, %1641 : tensor<8x4096x9x128xbf16>
    %cst_240 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1720 = stablehlo.reduce(%1719 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<8x4096x9x128xbf16>, tensor<bf16>) -> tensor<8x4096x9xbf16>
    %1721 = stablehlo.reshape %1720 : (tensor<8x4096x9xbf16>) -> tensor<8x4096x9x1xbf16>
    %1722 = stablehlo.negate %1721 : tensor<8x4096x9x1xbf16>
    %1723 = stablehlo.broadcast_in_dim %1638, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %1724 = stablehlo.divide %1716, %1723 : tensor<8x4096x9x128xbf16>
    %1725 = stablehlo.reshape %1724 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %1726 = stablehlo.transpose %1725, dims = [0, 2, 4, 3, 1] : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %1727 = stablehlo.dot_general %1726, %1536#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1728 = stablehlo.dot_general %1726, %1637, batching_dims = [0, 1] x [0, 1], contracting_dims = [3, 4] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x4096xbf16>
    %1729 = stablehlo.transpose %1728, dims = [0, 3, 1, 2] : (tensor<8x9x128x4096xbf16>) -> tensor<8x4096x9x128xbf16>
    %1730 = stablehlo.transpose %1722, dims = [0, 2, 1, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x9x4096x1xbf16>
    %1731 = stablehlo.reshape %1727 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %1732 = stablehlo.convert %1730 : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x1xf32>
    %cst_241 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1733 = stablehlo.reduce(%1732 init: %cst_241) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %1734 = stablehlo.broadcast_in_dim %1733, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x4096xf32>
    %1735 = stablehlo.convert %1734 : (tensor<8x9x4096x4096xf32>) -> tensor<8x9x4096x4096xbf16>
    %1736 = stablehlo.add %1731, %1735 : tensor<8x9x4096x4096xbf16>
    %1737 = stablehlo.multiply %1736, %1632 : tensor<8x9x4096x4096xbf16>
    %1738 = stablehlo.negate %1737 : tensor<8x9x4096x4096xbf16>
    %cst_242 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1739 = stablehlo.reduce(%1738 init: %cst_242) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %1740 = stablehlo.reshape %1739 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %cst_243 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1741 = stablehlo.reduce(%1740 init: %cst_243) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %1742 = stablehlo.divide %1741, %1628 : tensor<8x9x4096xbf16>
    %1743 = stablehlo.broadcast_in_dim %1742, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %1744 = stablehlo.multiply %1743, %1627 : tensor<8x9x4096x4096xbf16>
    %1745 = stablehlo.add %1737, %1744 : tensor<8x9x4096x4096xbf16>
    %1746 = stablehlo.reshape %1745 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1747 = call @_where_163(%1621#1, %1746) : (tensor<8x9x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1748 = stablehlo.transpose %1747, dims = [0, 1, 4, 3, 2] : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %1749 = stablehlo.dot_general %1748, %1536#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x4096x1x128xbf16>
    %1750 = stablehlo.transpose %1749, dims = [0, 2, 1, 3, 4] : (tensor<8x9x4096x1x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %1751 = stablehlo.dot_general %1748, %1601, batching_dims = [0, 1] x [0, 2], contracting_dims = [3, 4] x [1, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x128xbf16>
    %1752 = stablehlo.transpose %1751, dims = [0, 2, 1, 3] : (tensor<8x9x4096x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1753 = stablehlo.reshape %1750 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1754 = stablehlo.custom_call @Sharding(%1729) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1755 = stablehlo.custom_call @Sharding(%1752) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1756 = stablehlo.custom_call @Sharding(%1753) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1757 = stablehlo.slice %1755 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %1758 = stablehlo.slice %1755 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %1759 = stablehlo.broadcast_in_dim %1598, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1760 = stablehlo.multiply %1758, %1759 : tensor<8x4096x9x64xbf16>
    %1761 = stablehlo.broadcast_in_dim %1600, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1762 = stablehlo.multiply %1758, %1761 : tensor<8x4096x9x64xbf16>
    %1763 = stablehlo.negate %1757 : tensor<8x4096x9x64xbf16>
    %1764 = stablehlo.broadcast_in_dim %1598, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1765 = stablehlo.multiply %1763, %1764 : tensor<8x4096x9x64xbf16>
    %1766 = stablehlo.add %1762, %1765 : tensor<8x4096x9x64xbf16>
    %1767 = stablehlo.broadcast_in_dim %1600, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1768 = stablehlo.multiply %1757, %1767 : tensor<8x4096x9x64xbf16>
    %1769 = stablehlo.add %1760, %1768 : tensor<8x4096x9x64xbf16>
    %1770 = stablehlo.concatenate %1769, %1766, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %1771 = stablehlo.slice %1756 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %1772 = stablehlo.slice %1756 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %1773 = stablehlo.broadcast_in_dim %1578, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1774 = stablehlo.multiply %1772, %1773 : tensor<8x4096x9x64xbf16>
    %1775 = stablehlo.broadcast_in_dim %1580, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1776 = stablehlo.multiply %1772, %1775 : tensor<8x4096x9x64xbf16>
    %1777 = stablehlo.negate %1771 : tensor<8x4096x9x64xbf16>
    %1778 = stablehlo.broadcast_in_dim %1578, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1779 = stablehlo.multiply %1777, %1778 : tensor<8x4096x9x64xbf16>
    %1780 = stablehlo.add %1776, %1779 : tensor<8x4096x9x64xbf16>
    %1781 = stablehlo.broadcast_in_dim %1580, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %1782 = stablehlo.multiply %1771, %1781 : tensor<8x4096x9x64xbf16>
    %1783 = stablehlo.add %1774, %1782 : tensor<8x4096x9x64xbf16>
    %1784 = stablehlo.concatenate %1783, %1780, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %1785 = stablehlo.dot_general %1754, %1560, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %1786 = stablehlo.transpose %1785, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %1787 = stablehlo.dot_general %1754, %1536#14, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %1788 = stablehlo.dot_general %1770, %1560, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %1789 = stablehlo.transpose %1788, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %1790 = stablehlo.dot_general %1770, %1536#11, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %1791 = stablehlo.add %1787, %1790 : tensor<8x4096x576xbf16>
    %1792 = stablehlo.dot_general %1784, %1559, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %1793 = stablehlo.transpose %1792, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %1794 = stablehlo.dot_general %1784, %1536#13, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %1795 = stablehlo.custom_call @Sharding(%1791) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1796 = stablehlo.custom_call @Sharding(%1794) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1797 = stablehlo.add %1795, %1796 : tensor<8x4096x576xbf16>
    %1798 = stablehlo.custom_call @Sharding(%1797) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1799 = stablehlo.multiply %1554, %1798 : tensor<8x4096x576xbf16>
    %cst_244 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1800 = stablehlo.reduce(%1799 init: %cst_244) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %1801 = stablehlo.reshape %1800 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1802 = stablehlo.broadcast_in_dim %1555, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1803 = stablehlo.multiply %1798, %1802 : tensor<8x4096x576xbf16>
    %cst_245 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1804 = stablehlo.reduce(%1801 init: %cst_245) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %1805 = stablehlo.convert %1803 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1806 = stablehlo.multiply %1538, %1805 : tensor<8x4096x576xf32>
    %cst_246 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1807 = stablehlo.reduce(%1806 init: %cst_246) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1808 = stablehlo.reshape %1807 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %1809 = stablehlo.broadcast_in_dim %1548, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1810 = stablehlo.multiply %1805, %1809 : tensor<8x4096x576xf32>
    %1811 = stablehlo.multiply %1808, %1551 : tensor<8x4096x1xf32>
    %cst_247 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1812 = stablehlo.broadcast_in_dim %cst_247, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1813 = stablehlo.divide %1811, %1812 : tensor<8x4096x1xf32>
    %cst_248 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1814 = stablehlo.reduce(%1813 init: %cst_248) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1815 = stablehlo.broadcast_in_dim %1814, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %1816 = stablehlo.multiply %1815, %1541 : tensor<8x4096x576xf32>
    %1817 = stablehlo.add %1810, %1816 : tensor<8x4096x576xf32>
    %1818 = stablehlo.convert %1817 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1819 = stablehlo.add %1711, %1818 : tensor<8x4096x576xbf16>
    %1820 = stablehlo.custom_call @Sharding(%1819) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1821:19 = stablehlo.optimization_barrier %752, %753, %754, %795, %815#1, %817, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %660, %arg203, %arg202, %1820 : tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096xi32>, tensor<8x4096xi32>, tensor<8x4096x576xbf16>
    %1822 = stablehlo.custom_call @Sharding(%1821#15) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1823 = stablehlo.convert %1822 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1824 = stablehlo.multiply %1823, %1823 : tensor<8x4096x576xf32>
    %cst_249 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1825 = stablehlo.broadcast_in_dim %cst_249, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %1826 = stablehlo.multiply %1825, %1823 : tensor<8x4096x576xf32>
    %cst_250 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1827 = stablehlo.reduce(%1824 init: %cst_250) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1828 = stablehlo.broadcast_in_dim %1827, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_251 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1829 = stablehlo.broadcast_in_dim %cst_251, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1830 = stablehlo.divide %1828, %1829 : tensor<8x4096x1xf32>
    %cst_252 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1831 = stablehlo.broadcast_in_dim %cst_252, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1832 = stablehlo.add %1830, %1831 : tensor<8x4096x1xf32>
    %1833 = stablehlo.rsqrt %1832 : tensor<8x4096x1xf32>
    %1834 = stablehlo.divide %1833, %1832 : tensor<8x4096x1xf32>
    %cst_253 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1835 = stablehlo.broadcast_in_dim %cst_253, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1836 = stablehlo.multiply %1835, %1834 : tensor<8x4096x1xf32>
    %1837 = stablehlo.broadcast_in_dim %1833, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1838 = stablehlo.multiply %1823, %1837 : tensor<8x4096x576xf32>
    %1839 = stablehlo.convert %1838 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1840 = stablehlo.broadcast_in_dim %1821#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1841 = stablehlo.broadcast_in_dim %1840, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1842 = stablehlo.multiply %1839, %1841 : tensor<8x4096x576xbf16>
    %1843 = stablehlo.custom_call @Sharding(%1842) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1844 = stablehlo.custom_call @Sharding(%1843) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1845 = stablehlo.custom_call @Sharding(%1843) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1846 = stablehlo.broadcast_in_dim %1821#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %1847 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_254 = stablehlo.constant dense<2> : tensor<i32>
    %1848 = stablehlo.broadcast_in_dim %c_254, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %1849 = stablehlo.multiply %1848, %1847 : tensor<64xi32>
    %1850 = stablehlo.convert %1849 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_255 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1851 = stablehlo.broadcast_in_dim %cst_255, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1852 = stablehlo.divide %1850, %1851 : tensor<64xf32>
    %cst_256 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %1853 = stablehlo.broadcast_in_dim %cst_256, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1854 = stablehlo.power %1853, %1852 : tensor<64xf32>
    %cst_257 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1855 = stablehlo.broadcast_in_dim %cst_257, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1856 = stablehlo.multiply %1855, %1854 : tensor<64xf32>
    %1857 = stablehlo.convert %1846 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %1858 = stablehlo.broadcast_in_dim %1856, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %1859 = stablehlo.broadcast_in_dim %1857, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %1860 = stablehlo.broadcast_in_dim %1858, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %1861 = stablehlo.divide %1859, %1860 : tensor<8x4096x1x64xf32>
    %1862 = stablehlo.sine %1861 : tensor<8x4096x1x64xf32>
    %1863 = stablehlo.convert %1862 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1864 = stablehlo.cosine %1861 : tensor<8x4096x1x64xf32>
    %1865 = stablehlo.convert %1864 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1866 = stablehlo.broadcast_in_dim %1821#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %1867 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_258 = stablehlo.constant dense<2> : tensor<i32>
    %1868 = stablehlo.broadcast_in_dim %c_258, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %1869 = stablehlo.multiply %1868, %1867 : tensor<64xi32>
    %1870 = stablehlo.convert %1869 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_259 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1871 = stablehlo.broadcast_in_dim %cst_259, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1872 = stablehlo.divide %1870, %1871 : tensor<64xf32>
    %cst_260 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %1873 = stablehlo.broadcast_in_dim %cst_260, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1874 = stablehlo.power %1873, %1872 : tensor<64xf32>
    %cst_261 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1875 = stablehlo.broadcast_in_dim %cst_261, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1876 = stablehlo.multiply %1875, %1874 : tensor<64xf32>
    %1877 = stablehlo.convert %1866 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %1878 = stablehlo.broadcast_in_dim %1876, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %1879 = stablehlo.broadcast_in_dim %1877, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %1880 = stablehlo.broadcast_in_dim %1878, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %1881 = stablehlo.divide %1879, %1880 : tensor<8x4096x1x64xf32>
    %1882 = stablehlo.sine %1881 : tensor<8x4096x1x64xf32>
    %1883 = stablehlo.convert %1882 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1884 = stablehlo.cosine %1881 : tensor<8x4096x1x64xf32>
    %1885 = stablehlo.convert %1884 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %1886 = stablehlo.reshape %1821#0 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %1887 = stablehlo.dot_general %1821#1, %1886, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %1888 = stablehlo.transpose %1887, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1889 = stablehlo.broadcast_in_dim %1821#16, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %1890 = stablehlo.broadcast_in_dim %1821#16, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %1891 = stablehlo.broadcast_in_dim %1889, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %1892 = stablehlo.broadcast_in_dim %1890, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %1893 = stablehlo.compare  EQ, %1891, %1892,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %1894 = stablehlo.broadcast_in_dim %1893, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %1895 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %1896 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_262 = stablehlo.constant dense<0> : tensor<i32>
    %1897 = stablehlo.broadcast_in_dim %c_262, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %1898 = stablehlo.add %1895, %1897 : tensor<4096x4096xi32>
    %1899 = stablehlo.compare  LE, %1896, %1898,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %1900 = stablehlo.broadcast_in_dim %1899, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %1901 = stablehlo.broadcast_in_dim %1900, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %1902 = stablehlo.and %1894, %1901 : tensor<8x1x1x4096x4096xi1>
    %cst_263 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_264 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %1903 = call @_where_115(%1902, %cst_263, %cst_264) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_265 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %1904 = stablehlo.broadcast_in_dim %cst_265, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %1905 = stablehlo.compare  GE, %1903, %1904,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_266 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %1906:2 = call @_where_116(%1905, %1888, %cst_266) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> (tensor<8x9x1x4096x4096xbf16>, tensor<8x9x1x4096x4096xi1>)
    %1907 = stablehlo.reshape %1906#0 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_267 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1908 = stablehlo.reduce(%1907 init: %cst_267) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %1909 = stablehlo.reshape %1908 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %1910 = stablehlo.broadcast_in_dim %1909, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %1911 = stablehlo.compare  EQ, %1907, %1910,  FLOAT : (tensor<8x9x4096x4096xbf16>, tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xi1>
    %1912 = stablehlo.convert %1911 : (tensor<8x9x4096x4096xi1>) -> tensor<8x9x4096x4096xbf16>
    %cst_268 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1913 = stablehlo.reduce(%1912 init: %cst_268) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %1914 = stablehlo.broadcast_in_dim %1908, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %1915 = stablehlo.broadcast_in_dim %1914, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %1916 = stablehlo.subtract %1907, %1915 : tensor<8x9x4096x4096xbf16>
    %1917 = stablehlo.exponential %1916 : tensor<8x9x4096x4096xbf16>
    %1918 = stablehlo.convert %1917 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_269 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1919 = stablehlo.reduce(%1918 init: %cst_269) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %1920 = stablehlo.broadcast_in_dim %1919, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %1921 = stablehlo.convert %1920 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %1922 = stablehlo.reshape %1917 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1923 = stablehlo.transpose %1921, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %1924 = stablehlo.dot_general %1821#2, %1922, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %1925 = stablehlo.transpose %1924, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %1926 = stablehlo.reshape %1925 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1927 = stablehlo.broadcast_in_dim %1923, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %1928 = stablehlo.divide %1926, %1927 : tensor<8x4096x9x128xbf16>
    %1929 = stablehlo.multiply %1923, %1923 : tensor<8x4096x9x1xbf16>
    %cst_270 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1930 = stablehlo.broadcast_in_dim %cst_270, dims = [] : (tensor<bf16>) -> tensor<8x4096x9x1xbf16>
    %1931 = stablehlo.divide %1930, %1929 : tensor<8x4096x9x1xbf16>
    %1932 = stablehlo.custom_call @Sharding(%1928) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %1933 = stablehlo.custom_call @Sharding(%1821#3) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1934 = stablehlo.add %1822, %1933 : tensor<8x4096x576xbf16>
    %1935 = stablehlo.convert %1934 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1936 = stablehlo.multiply %1935, %1935 : tensor<8x4096x576xf32>
    %cst_271 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1937 = stablehlo.broadcast_in_dim %cst_271, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %1938 = stablehlo.multiply %1937, %1935 : tensor<8x4096x576xf32>
    %cst_272 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1939 = stablehlo.reduce(%1936 init: %cst_272) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1940 = stablehlo.broadcast_in_dim %1939, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_273 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1941 = stablehlo.broadcast_in_dim %cst_273, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1942 = stablehlo.divide %1940, %1941 : tensor<8x4096x1xf32>
    %cst_274 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1943 = stablehlo.broadcast_in_dim %cst_274, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1944 = stablehlo.add %1942, %1943 : tensor<8x4096x1xf32>
    %1945 = stablehlo.rsqrt %1944 : tensor<8x4096x1xf32>
    %1946 = stablehlo.divide %1945, %1944 : tensor<8x4096x1xf32>
    %cst_275 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1947 = stablehlo.broadcast_in_dim %cst_275, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1948 = stablehlo.multiply %1947, %1946 : tensor<8x4096x1xf32>
    %1949 = stablehlo.broadcast_in_dim %1945, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1950 = stablehlo.multiply %1935, %1949 : tensor<8x4096x576xf32>
    %1951 = stablehlo.convert %1950 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1952 = stablehlo.broadcast_in_dim %1821#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1953 = stablehlo.broadcast_in_dim %1952, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1954 = stablehlo.multiply %1951, %1953 : tensor<8x4096x576xbf16>
    %1955 = stablehlo.custom_call @Sharding(%1954) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1956:3 = call @silu_124(%1821#4) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %1957 = stablehlo.multiply %1956#0, %1821#5 : tensor<8x4096x2016xbf16>
    %1958 = stablehlo.custom_call @Sharding(%1957) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %1959 = stablehlo.custom_call @Sharding(%1821#18) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1960 = stablehlo.custom_call @Sharding(%1959) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1961 = stablehlo.dot_general %1960, %1958, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %1962 = stablehlo.transpose %1961, dims = [1, 0] : (tensor<576x2016xbf16>) -> tensor<2016x576xbf16>
    %1963 = stablehlo.dot_general %1960, %1821#8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x2016xbf16>
    %1964 = stablehlo.custom_call @Sharding(%1963) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %1965 = stablehlo.multiply %1956#0, %1964 : tensor<8x4096x2016xbf16>
    %1966 = stablehlo.multiply %1964, %1821#5 : tensor<8x4096x2016xbf16>
    %1967 = stablehlo.dot_general %1965, %1955, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %1968 = stablehlo.transpose %1967, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %1969 = stablehlo.dot_general %1965, %1821#7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %1970 = call @silu_133(%1956#1, %1821#4, %1956#2, %1966) : (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %1971 = stablehlo.dot_general %1970, %1955, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %1972 = stablehlo.transpose %1971, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %1973 = stablehlo.dot_general %1970, %1821#6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %1974 = stablehlo.add %1969, %1973 : tensor<8x4096x576xbf16>
    %1975 = stablehlo.custom_call @Sharding(%1974) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1976 = stablehlo.multiply %1951, %1975 : tensor<8x4096x576xbf16>
    %cst_276 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1977 = stablehlo.reduce(%1976 init: %cst_276) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %1978 = stablehlo.reshape %1977 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %1979 = stablehlo.broadcast_in_dim %1952, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %1980 = stablehlo.multiply %1975, %1979 : tensor<8x4096x576xbf16>
    %cst_277 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1981 = stablehlo.reduce(%1978 init: %cst_277) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %1982 = stablehlo.convert %1980 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %1983 = stablehlo.multiply %1935, %1982 : tensor<8x4096x576xf32>
    %cst_278 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1984 = stablehlo.reduce(%1983 init: %cst_278) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1985 = stablehlo.reshape %1984 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %1986 = stablehlo.broadcast_in_dim %1945, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %1987 = stablehlo.multiply %1982, %1986 : tensor<8x4096x576xf32>
    %1988 = stablehlo.multiply %1985, %1948 : tensor<8x4096x1xf32>
    %cst_279 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %1989 = stablehlo.broadcast_in_dim %cst_279, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %1990 = stablehlo.divide %1988, %1989 : tensor<8x4096x1xf32>
    %cst_280 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1991 = stablehlo.reduce(%1990 init: %cst_280) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %1992 = stablehlo.broadcast_in_dim %1991, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %1993 = stablehlo.multiply %1992, %1938 : tensor<8x4096x576xf32>
    %1994 = stablehlo.add %1987, %1993 : tensor<8x4096x576xf32>
    %1995 = stablehlo.convert %1994 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %1996 = stablehlo.add %1959, %1995 : tensor<8x4096x576xbf16>
    %1997 = stablehlo.custom_call @Sharding(%1996) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %1998 = stablehlo.dot_general %1997, %1932, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1999 = stablehlo.transpose %1998, dims = [1, 2, 0] : (tensor<576x9x128xbf16>) -> tensor<9x128x576xbf16>
    %2000 = stablehlo.dot_general %1997, %1821#12, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x9x128xbf16>
    %2001 = stablehlo.custom_call @Sharding(%2000) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2002 = stablehlo.broadcast_in_dim %1931, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %2003 = stablehlo.multiply %2001, %2002 : tensor<8x4096x9x128xbf16>
    %2004 = stablehlo.multiply %2003, %1926 : tensor<8x4096x9x128xbf16>
    %cst_281 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2005 = stablehlo.reduce(%2004 init: %cst_281) applies stablehlo.add across dimensions = [3] : (tensor<8x4096x9x128xbf16>, tensor<bf16>) -> tensor<8x4096x9xbf16>
    %2006 = stablehlo.reshape %2005 : (tensor<8x4096x9xbf16>) -> tensor<8x4096x9x1xbf16>
    %2007 = stablehlo.negate %2006 : tensor<8x4096x9x1xbf16>
    %2008 = stablehlo.broadcast_in_dim %1923, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %2009 = stablehlo.divide %2001, %2008 : tensor<8x4096x9x128xbf16>
    %2010 = stablehlo.reshape %2009 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2011 = stablehlo.transpose %2010, dims = [0, 2, 4, 3, 1] : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %2012 = stablehlo.dot_general %2011, %1821#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2013 = stablehlo.dot_general %2011, %1922, batching_dims = [0, 1] x [0, 1], contracting_dims = [3, 4] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x4096xbf16>
    %2014 = stablehlo.transpose %2013, dims = [0, 3, 1, 2] : (tensor<8x9x128x4096xbf16>) -> tensor<8x4096x9x128xbf16>
    %2015 = stablehlo.transpose %2007, dims = [0, 2, 1, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x9x4096x1xbf16>
    %2016 = stablehlo.reshape %2012 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2017 = stablehlo.convert %2015 : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x1xf32>
    %cst_282 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2018 = stablehlo.reduce(%2017 init: %cst_282) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %2019 = stablehlo.broadcast_in_dim %2018, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x4096xf32>
    %2020 = stablehlo.convert %2019 : (tensor<8x9x4096x4096xf32>) -> tensor<8x9x4096x4096xbf16>
    %2021 = stablehlo.add %2016, %2020 : tensor<8x9x4096x4096xbf16>
    %2022 = stablehlo.multiply %2021, %1917 : tensor<8x9x4096x4096xbf16>
    %2023 = stablehlo.negate %2022 : tensor<8x9x4096x4096xbf16>
    %cst_283 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2024 = stablehlo.reduce(%2023 init: %cst_283) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2025 = stablehlo.reshape %2024 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %cst_284 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2026 = stablehlo.reduce(%2025 init: %cst_284) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2027 = stablehlo.divide %2026, %1913 : tensor<8x9x4096xbf16>
    %2028 = stablehlo.broadcast_in_dim %2027, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2029 = stablehlo.multiply %2028, %1912 : tensor<8x9x4096x4096xbf16>
    %2030 = stablehlo.add %2022, %2029 : tensor<8x9x4096x4096xbf16>
    %2031 = stablehlo.reshape %2030 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2032 = call @_where_163(%1906#1, %2031) : (tensor<8x9x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2033 = stablehlo.transpose %2032, dims = [0, 1, 4, 3, 2] : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %2034 = stablehlo.dot_general %2033, %1821#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x4096x1x128xbf16>
    %2035 = stablehlo.transpose %2034, dims = [0, 2, 1, 3, 4] : (tensor<8x9x4096x1x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2036 = stablehlo.dot_general %2033, %1886, batching_dims = [0, 1] x [0, 2], contracting_dims = [3, 4] x [1, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x128xbf16>
    %2037 = stablehlo.transpose %2036, dims = [0, 2, 1, 3] : (tensor<8x9x4096x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2038 = stablehlo.reshape %2035 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2039 = stablehlo.custom_call @Sharding(%2014) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2040 = stablehlo.custom_call @Sharding(%2037) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2041 = stablehlo.custom_call @Sharding(%2038) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2042 = stablehlo.slice %2040 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2043 = stablehlo.slice %2040 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2044 = stablehlo.broadcast_in_dim %1883, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2045 = stablehlo.multiply %2043, %2044 : tensor<8x4096x9x64xbf16>
    %2046 = stablehlo.broadcast_in_dim %1885, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2047 = stablehlo.multiply %2043, %2046 : tensor<8x4096x9x64xbf16>
    %2048 = stablehlo.negate %2042 : tensor<8x4096x9x64xbf16>
    %2049 = stablehlo.broadcast_in_dim %1883, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2050 = stablehlo.multiply %2048, %2049 : tensor<8x4096x9x64xbf16>
    %2051 = stablehlo.add %2047, %2050 : tensor<8x4096x9x64xbf16>
    %2052 = stablehlo.broadcast_in_dim %1885, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2053 = stablehlo.multiply %2042, %2052 : tensor<8x4096x9x64xbf16>
    %2054 = stablehlo.add %2045, %2053 : tensor<8x4096x9x64xbf16>
    %2055 = stablehlo.concatenate %2054, %2051, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %2056 = stablehlo.slice %2041 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2057 = stablehlo.slice %2041 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2058 = stablehlo.broadcast_in_dim %1863, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2059 = stablehlo.multiply %2057, %2058 : tensor<8x4096x9x64xbf16>
    %2060 = stablehlo.broadcast_in_dim %1865, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2061 = stablehlo.multiply %2057, %2060 : tensor<8x4096x9x64xbf16>
    %2062 = stablehlo.negate %2056 : tensor<8x4096x9x64xbf16>
    %2063 = stablehlo.broadcast_in_dim %1863, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2064 = stablehlo.multiply %2062, %2063 : tensor<8x4096x9x64xbf16>
    %2065 = stablehlo.add %2061, %2064 : tensor<8x4096x9x64xbf16>
    %2066 = stablehlo.broadcast_in_dim %1865, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2067 = stablehlo.multiply %2056, %2066 : tensor<8x4096x9x64xbf16>
    %2068 = stablehlo.add %2059, %2067 : tensor<8x4096x9x64xbf16>
    %2069 = stablehlo.concatenate %2068, %2065, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %2070 = stablehlo.dot_general %2039, %1845, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2071 = stablehlo.transpose %2070, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %2072 = stablehlo.dot_general %2039, %1821#14, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %2073 = stablehlo.dot_general %2055, %1845, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2074 = stablehlo.transpose %2073, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %2075 = stablehlo.dot_general %2055, %1821#11, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %2076 = stablehlo.add %2072, %2075 : tensor<8x4096x576xbf16>
    %2077 = stablehlo.dot_general %2069, %1844, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2078 = stablehlo.transpose %2077, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %2079 = stablehlo.dot_general %2069, %1821#13, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %2080 = stablehlo.custom_call @Sharding(%2076) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2081 = stablehlo.custom_call @Sharding(%2079) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2082 = stablehlo.add %2080, %2081 : tensor<8x4096x576xbf16>
    %2083 = stablehlo.custom_call @Sharding(%2082) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2084 = stablehlo.multiply %1839, %2083 : tensor<8x4096x576xbf16>
    %cst_285 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2085 = stablehlo.reduce(%2084 init: %cst_285) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2086 = stablehlo.reshape %2085 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2087 = stablehlo.broadcast_in_dim %1840, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2088 = stablehlo.multiply %2083, %2087 : tensor<8x4096x576xbf16>
    %cst_286 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2089 = stablehlo.reduce(%2086 init: %cst_286) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2090 = stablehlo.convert %2088 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2091 = stablehlo.multiply %1823, %2090 : tensor<8x4096x576xf32>
    %cst_287 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2092 = stablehlo.reduce(%2091 init: %cst_287) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2093 = stablehlo.reshape %2092 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %2094 = stablehlo.broadcast_in_dim %1833, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2095 = stablehlo.multiply %2090, %2094 : tensor<8x4096x576xf32>
    %2096 = stablehlo.multiply %2093, %1836 : tensor<8x4096x1xf32>
    %cst_288 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2097 = stablehlo.broadcast_in_dim %cst_288, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2098 = stablehlo.divide %2096, %2097 : tensor<8x4096x1xf32>
    %cst_289 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2099 = stablehlo.reduce(%2098 init: %cst_289) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2100 = stablehlo.broadcast_in_dim %2099, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %2101 = stablehlo.multiply %2100, %1826 : tensor<8x4096x576xf32>
    %2102 = stablehlo.add %2095, %2101 : tensor<8x4096x576xf32>
    %2103 = stablehlo.convert %2102 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2104 = stablehlo.add %1996, %2103 : tensor<8x4096x576xbf16>
    %2105 = stablehlo.custom_call @Sharding(%2104) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2106:19 = stablehlo.optimization_barrier %589, %590, %591, %632, %652#1, %654, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %497, %arg203, %arg202, %2105 : tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096xi32>, tensor<8x4096xi32>, tensor<8x4096x576xbf16>
    %2107 = stablehlo.custom_call @Sharding(%2106#15) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2108 = stablehlo.convert %2107 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2109 = stablehlo.multiply %2108, %2108 : tensor<8x4096x576xf32>
    %cst_290 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %2110 = stablehlo.broadcast_in_dim %cst_290, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %2111 = stablehlo.multiply %2110, %2108 : tensor<8x4096x576xf32>
    %cst_291 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2112 = stablehlo.reduce(%2109 init: %cst_291) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2113 = stablehlo.broadcast_in_dim %2112, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_292 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2114 = stablehlo.broadcast_in_dim %cst_292, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2115 = stablehlo.divide %2113, %2114 : tensor<8x4096x1xf32>
    %cst_293 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2116 = stablehlo.broadcast_in_dim %cst_293, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2117 = stablehlo.add %2115, %2116 : tensor<8x4096x1xf32>
    %2118 = stablehlo.rsqrt %2117 : tensor<8x4096x1xf32>
    %2119 = stablehlo.divide %2118, %2117 : tensor<8x4096x1xf32>
    %cst_294 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %2120 = stablehlo.broadcast_in_dim %cst_294, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2121 = stablehlo.multiply %2120, %2119 : tensor<8x4096x1xf32>
    %2122 = stablehlo.broadcast_in_dim %2118, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2123 = stablehlo.multiply %2108, %2122 : tensor<8x4096x576xf32>
    %2124 = stablehlo.convert %2123 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2125 = stablehlo.broadcast_in_dim %2106#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2126 = stablehlo.broadcast_in_dim %2125, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2127 = stablehlo.multiply %2124, %2126 : tensor<8x4096x576xbf16>
    %2128 = stablehlo.custom_call @Sharding(%2127) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2129 = stablehlo.custom_call @Sharding(%2128) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2130 = stablehlo.custom_call @Sharding(%2128) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2131 = stablehlo.broadcast_in_dim %2106#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %2132 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_295 = stablehlo.constant dense<2> : tensor<i32>
    %2133 = stablehlo.broadcast_in_dim %c_295, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %2134 = stablehlo.multiply %2133, %2132 : tensor<64xi32>
    %2135 = stablehlo.convert %2134 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_296 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2136 = stablehlo.broadcast_in_dim %cst_296, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2137 = stablehlo.divide %2135, %2136 : tensor<64xf32>
    %cst_297 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %2138 = stablehlo.broadcast_in_dim %cst_297, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2139 = stablehlo.power %2138, %2137 : tensor<64xf32>
    %cst_298 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2140 = stablehlo.broadcast_in_dim %cst_298, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2141 = stablehlo.multiply %2140, %2139 : tensor<64xf32>
    %2142 = stablehlo.convert %2131 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %2143 = stablehlo.broadcast_in_dim %2141, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2144 = stablehlo.broadcast_in_dim %2142, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %2145 = stablehlo.broadcast_in_dim %2143, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %2146 = stablehlo.divide %2144, %2145 : tensor<8x4096x1x64xf32>
    %2147 = stablehlo.sine %2146 : tensor<8x4096x1x64xf32>
    %2148 = stablehlo.convert %2147 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %2149 = stablehlo.cosine %2146 : tensor<8x4096x1x64xf32>
    %2150 = stablehlo.convert %2149 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %2151 = stablehlo.broadcast_in_dim %2106#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %2152 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_299 = stablehlo.constant dense<2> : tensor<i32>
    %2153 = stablehlo.broadcast_in_dim %c_299, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %2154 = stablehlo.multiply %2153, %2152 : tensor<64xi32>
    %2155 = stablehlo.convert %2154 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_300 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2156 = stablehlo.broadcast_in_dim %cst_300, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2157 = stablehlo.divide %2155, %2156 : tensor<64xf32>
    %cst_301 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %2158 = stablehlo.broadcast_in_dim %cst_301, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2159 = stablehlo.power %2158, %2157 : tensor<64xf32>
    %cst_302 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2160 = stablehlo.broadcast_in_dim %cst_302, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2161 = stablehlo.multiply %2160, %2159 : tensor<64xf32>
    %2162 = stablehlo.convert %2151 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %2163 = stablehlo.broadcast_in_dim %2161, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2164 = stablehlo.broadcast_in_dim %2162, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %2165 = stablehlo.broadcast_in_dim %2163, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %2166 = stablehlo.divide %2164, %2165 : tensor<8x4096x1x64xf32>
    %2167 = stablehlo.sine %2166 : tensor<8x4096x1x64xf32>
    %2168 = stablehlo.convert %2167 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %2169 = stablehlo.cosine %2166 : tensor<8x4096x1x64xf32>
    %2170 = stablehlo.convert %2169 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %2171 = stablehlo.reshape %2106#0 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2172 = stablehlo.dot_general %2106#1, %2171, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %2173 = stablehlo.transpose %2172, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2174 = stablehlo.broadcast_in_dim %2106#16, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %2175 = stablehlo.broadcast_in_dim %2106#16, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %2176 = stablehlo.broadcast_in_dim %2174, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %2177 = stablehlo.broadcast_in_dim %2175, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %2178 = stablehlo.compare  EQ, %2176, %2177,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %2179 = stablehlo.broadcast_in_dim %2178, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %2180 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %2181 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_303 = stablehlo.constant dense<0> : tensor<i32>
    %2182 = stablehlo.broadcast_in_dim %c_303, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %2183 = stablehlo.add %2180, %2182 : tensor<4096x4096xi32>
    %2184 = stablehlo.compare  LE, %2181, %2183,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %2185 = stablehlo.broadcast_in_dim %2184, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %2186 = stablehlo.broadcast_in_dim %2185, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %2187 = stablehlo.and %2179, %2186 : tensor<8x1x1x4096x4096xi1>
    %cst_304 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_305 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %2188 = call @_where_115(%2187, %cst_304, %cst_305) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_306 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %2189 = stablehlo.broadcast_in_dim %cst_306, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %2190 = stablehlo.compare  GE, %2188, %2189,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_307 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %2191:2 = call @_where_116(%2190, %2173, %cst_307) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> (tensor<8x9x1x4096x4096xbf16>, tensor<8x9x1x4096x4096xi1>)
    %2192 = stablehlo.reshape %2191#0 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_308 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2193 = stablehlo.reduce(%2192 init: %cst_308) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2194 = stablehlo.reshape %2193 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %2195 = stablehlo.broadcast_in_dim %2194, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2196 = stablehlo.compare  EQ, %2192, %2195,  FLOAT : (tensor<8x9x4096x4096xbf16>, tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xi1>
    %2197 = stablehlo.convert %2196 : (tensor<8x9x4096x4096xi1>) -> tensor<8x9x4096x4096xbf16>
    %cst_309 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2198 = stablehlo.reduce(%2197 init: %cst_309) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2199 = stablehlo.broadcast_in_dim %2193, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %2200 = stablehlo.broadcast_in_dim %2199, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2201 = stablehlo.subtract %2192, %2200 : tensor<8x9x4096x4096xbf16>
    %2202 = stablehlo.exponential %2201 : tensor<8x9x4096x4096xbf16>
    %2203 = stablehlo.convert %2202 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_310 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2204 = stablehlo.reduce(%2203 init: %cst_310) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %2205 = stablehlo.broadcast_in_dim %2204, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %2206 = stablehlo.convert %2205 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %2207 = stablehlo.reshape %2202 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2208 = stablehlo.transpose %2206, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %2209 = stablehlo.dot_general %2106#2, %2207, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %2210 = stablehlo.transpose %2209, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2211 = stablehlo.reshape %2210 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2212 = stablehlo.broadcast_in_dim %2208, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %2213 = stablehlo.divide %2211, %2212 : tensor<8x4096x9x128xbf16>
    %2214 = stablehlo.multiply %2208, %2208 : tensor<8x4096x9x1xbf16>
    %cst_311 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2215 = stablehlo.broadcast_in_dim %cst_311, dims = [] : (tensor<bf16>) -> tensor<8x4096x9x1xbf16>
    %2216 = stablehlo.divide %2215, %2214 : tensor<8x4096x9x1xbf16>
    %2217 = stablehlo.custom_call @Sharding(%2213) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2218 = stablehlo.custom_call @Sharding(%2106#3) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2219 = stablehlo.add %2107, %2218 : tensor<8x4096x576xbf16>
    %2220 = stablehlo.convert %2219 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2221 = stablehlo.multiply %2220, %2220 : tensor<8x4096x576xf32>
    %cst_312 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %2222 = stablehlo.broadcast_in_dim %cst_312, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %2223 = stablehlo.multiply %2222, %2220 : tensor<8x4096x576xf32>
    %cst_313 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2224 = stablehlo.reduce(%2221 init: %cst_313) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2225 = stablehlo.broadcast_in_dim %2224, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_314 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2226 = stablehlo.broadcast_in_dim %cst_314, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2227 = stablehlo.divide %2225, %2226 : tensor<8x4096x1xf32>
    %cst_315 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2228 = stablehlo.broadcast_in_dim %cst_315, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2229 = stablehlo.add %2227, %2228 : tensor<8x4096x1xf32>
    %2230 = stablehlo.rsqrt %2229 : tensor<8x4096x1xf32>
    %2231 = stablehlo.divide %2230, %2229 : tensor<8x4096x1xf32>
    %cst_316 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %2232 = stablehlo.broadcast_in_dim %cst_316, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2233 = stablehlo.multiply %2232, %2231 : tensor<8x4096x1xf32>
    %2234 = stablehlo.broadcast_in_dim %2230, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2235 = stablehlo.multiply %2220, %2234 : tensor<8x4096x576xf32>
    %2236 = stablehlo.convert %2235 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2237 = stablehlo.broadcast_in_dim %2106#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2238 = stablehlo.broadcast_in_dim %2237, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2239 = stablehlo.multiply %2236, %2238 : tensor<8x4096x576xbf16>
    %2240 = stablehlo.custom_call @Sharding(%2239) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2241:3 = call @silu_124(%2106#4) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %2242 = stablehlo.multiply %2241#0, %2106#5 : tensor<8x4096x2016xbf16>
    %2243 = stablehlo.custom_call @Sharding(%2242) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %2244 = stablehlo.custom_call @Sharding(%2106#18) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2245 = stablehlo.custom_call @Sharding(%2244) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2246 = stablehlo.dot_general %2245, %2243, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2247 = stablehlo.transpose %2246, dims = [1, 0] : (tensor<576x2016xbf16>) -> tensor<2016x576xbf16>
    %2248 = stablehlo.dot_general %2245, %2106#8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x2016xbf16>
    %2249 = stablehlo.custom_call @Sharding(%2248) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %2250 = stablehlo.multiply %2241#0, %2249 : tensor<8x4096x2016xbf16>
    %2251 = stablehlo.multiply %2249, %2106#5 : tensor<8x4096x2016xbf16>
    %2252 = stablehlo.dot_general %2250, %2240, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %2253 = stablehlo.transpose %2252, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %2254 = stablehlo.dot_general %2250, %2106#7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %2255 = call @silu_133(%2241#1, %2106#4, %2241#2, %2251) : (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %2256 = stablehlo.dot_general %2255, %2240, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %2257 = stablehlo.transpose %2256, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %2258 = stablehlo.dot_general %2255, %2106#6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %2259 = stablehlo.add %2254, %2258 : tensor<8x4096x576xbf16>
    %2260 = stablehlo.custom_call @Sharding(%2259) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2261 = stablehlo.multiply %2236, %2260 : tensor<8x4096x576xbf16>
    %cst_317 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2262 = stablehlo.reduce(%2261 init: %cst_317) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2263 = stablehlo.reshape %2262 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2264 = stablehlo.broadcast_in_dim %2237, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2265 = stablehlo.multiply %2260, %2264 : tensor<8x4096x576xbf16>
    %cst_318 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2266 = stablehlo.reduce(%2263 init: %cst_318) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2267 = stablehlo.convert %2265 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2268 = stablehlo.multiply %2220, %2267 : tensor<8x4096x576xf32>
    %cst_319 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2269 = stablehlo.reduce(%2268 init: %cst_319) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2270 = stablehlo.reshape %2269 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %2271 = stablehlo.broadcast_in_dim %2230, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2272 = stablehlo.multiply %2267, %2271 : tensor<8x4096x576xf32>
    %2273 = stablehlo.multiply %2270, %2233 : tensor<8x4096x1xf32>
    %cst_320 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2274 = stablehlo.broadcast_in_dim %cst_320, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2275 = stablehlo.divide %2273, %2274 : tensor<8x4096x1xf32>
    %cst_321 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2276 = stablehlo.reduce(%2275 init: %cst_321) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2277 = stablehlo.broadcast_in_dim %2276, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %2278 = stablehlo.multiply %2277, %2223 : tensor<8x4096x576xf32>
    %2279 = stablehlo.add %2272, %2278 : tensor<8x4096x576xf32>
    %2280 = stablehlo.convert %2279 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2281 = stablehlo.add %2244, %2280 : tensor<8x4096x576xbf16>
    %2282 = stablehlo.custom_call @Sharding(%2281) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2283 = stablehlo.dot_general %2282, %2217, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2284 = stablehlo.transpose %2283, dims = [1, 2, 0] : (tensor<576x9x128xbf16>) -> tensor<9x128x576xbf16>
    %2285 = stablehlo.dot_general %2282, %2106#12, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x9x128xbf16>
    %2286 = stablehlo.custom_call @Sharding(%2285) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2287 = stablehlo.broadcast_in_dim %2216, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %2288 = stablehlo.multiply %2286, %2287 : tensor<8x4096x9x128xbf16>
    %2289 = stablehlo.multiply %2288, %2211 : tensor<8x4096x9x128xbf16>
    %cst_322 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2290 = stablehlo.reduce(%2289 init: %cst_322) applies stablehlo.add across dimensions = [3] : (tensor<8x4096x9x128xbf16>, tensor<bf16>) -> tensor<8x4096x9xbf16>
    %2291 = stablehlo.reshape %2290 : (tensor<8x4096x9xbf16>) -> tensor<8x4096x9x1xbf16>
    %2292 = stablehlo.negate %2291 : tensor<8x4096x9x1xbf16>
    %2293 = stablehlo.broadcast_in_dim %2208, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %2294 = stablehlo.divide %2286, %2293 : tensor<8x4096x9x128xbf16>
    %2295 = stablehlo.reshape %2294 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2296 = stablehlo.transpose %2295, dims = [0, 2, 4, 3, 1] : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %2297 = stablehlo.dot_general %2296, %2106#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2298 = stablehlo.dot_general %2296, %2207, batching_dims = [0, 1] x [0, 1], contracting_dims = [3, 4] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x4096xbf16>
    %2299 = stablehlo.transpose %2298, dims = [0, 3, 1, 2] : (tensor<8x9x128x4096xbf16>) -> tensor<8x4096x9x128xbf16>
    %2300 = stablehlo.transpose %2292, dims = [0, 2, 1, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x9x4096x1xbf16>
    %2301 = stablehlo.reshape %2297 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2302 = stablehlo.convert %2300 : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x1xf32>
    %cst_323 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2303 = stablehlo.reduce(%2302 init: %cst_323) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %2304 = stablehlo.broadcast_in_dim %2303, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x4096xf32>
    %2305 = stablehlo.convert %2304 : (tensor<8x9x4096x4096xf32>) -> tensor<8x9x4096x4096xbf16>
    %2306 = stablehlo.add %2301, %2305 : tensor<8x9x4096x4096xbf16>
    %2307 = stablehlo.multiply %2306, %2202 : tensor<8x9x4096x4096xbf16>
    %2308 = stablehlo.negate %2307 : tensor<8x9x4096x4096xbf16>
    %cst_324 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2309 = stablehlo.reduce(%2308 init: %cst_324) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2310 = stablehlo.reshape %2309 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %cst_325 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2311 = stablehlo.reduce(%2310 init: %cst_325) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2312 = stablehlo.divide %2311, %2198 : tensor<8x9x4096xbf16>
    %2313 = stablehlo.broadcast_in_dim %2312, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2314 = stablehlo.multiply %2313, %2197 : tensor<8x9x4096x4096xbf16>
    %2315 = stablehlo.add %2307, %2314 : tensor<8x9x4096x4096xbf16>
    %2316 = stablehlo.reshape %2315 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2317 = call @_where_163(%2191#1, %2316) : (tensor<8x9x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2318 = stablehlo.transpose %2317, dims = [0, 1, 4, 3, 2] : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %2319 = stablehlo.dot_general %2318, %2106#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x4096x1x128xbf16>
    %2320 = stablehlo.transpose %2319, dims = [0, 2, 1, 3, 4] : (tensor<8x9x4096x1x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2321 = stablehlo.dot_general %2318, %2171, batching_dims = [0, 1] x [0, 2], contracting_dims = [3, 4] x [1, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x128xbf16>
    %2322 = stablehlo.transpose %2321, dims = [0, 2, 1, 3] : (tensor<8x9x4096x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2323 = stablehlo.reshape %2320 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2324 = stablehlo.custom_call @Sharding(%2299) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2325 = stablehlo.custom_call @Sharding(%2322) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2326 = stablehlo.custom_call @Sharding(%2323) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2327 = stablehlo.slice %2325 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2328 = stablehlo.slice %2325 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2329 = stablehlo.broadcast_in_dim %2168, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2330 = stablehlo.multiply %2328, %2329 : tensor<8x4096x9x64xbf16>
    %2331 = stablehlo.broadcast_in_dim %2170, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2332 = stablehlo.multiply %2328, %2331 : tensor<8x4096x9x64xbf16>
    %2333 = stablehlo.negate %2327 : tensor<8x4096x9x64xbf16>
    %2334 = stablehlo.broadcast_in_dim %2168, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2335 = stablehlo.multiply %2333, %2334 : tensor<8x4096x9x64xbf16>
    %2336 = stablehlo.add %2332, %2335 : tensor<8x4096x9x64xbf16>
    %2337 = stablehlo.broadcast_in_dim %2170, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2338 = stablehlo.multiply %2327, %2337 : tensor<8x4096x9x64xbf16>
    %2339 = stablehlo.add %2330, %2338 : tensor<8x4096x9x64xbf16>
    %2340 = stablehlo.concatenate %2339, %2336, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %2341 = stablehlo.slice %2326 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2342 = stablehlo.slice %2326 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2343 = stablehlo.broadcast_in_dim %2148, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2344 = stablehlo.multiply %2342, %2343 : tensor<8x4096x9x64xbf16>
    %2345 = stablehlo.broadcast_in_dim %2150, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2346 = stablehlo.multiply %2342, %2345 : tensor<8x4096x9x64xbf16>
    %2347 = stablehlo.negate %2341 : tensor<8x4096x9x64xbf16>
    %2348 = stablehlo.broadcast_in_dim %2148, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2349 = stablehlo.multiply %2347, %2348 : tensor<8x4096x9x64xbf16>
    %2350 = stablehlo.add %2346, %2349 : tensor<8x4096x9x64xbf16>
    %2351 = stablehlo.broadcast_in_dim %2150, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2352 = stablehlo.multiply %2341, %2351 : tensor<8x4096x9x64xbf16>
    %2353 = stablehlo.add %2344, %2352 : tensor<8x4096x9x64xbf16>
    %2354 = stablehlo.concatenate %2353, %2350, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %2355 = stablehlo.dot_general %2324, %2130, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2356 = stablehlo.transpose %2355, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %2357 = stablehlo.dot_general %2324, %2106#14, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %2358 = stablehlo.dot_general %2340, %2130, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2359 = stablehlo.transpose %2358, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %2360 = stablehlo.dot_general %2340, %2106#11, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %2361 = stablehlo.add %2357, %2360 : tensor<8x4096x576xbf16>
    %2362 = stablehlo.dot_general %2354, %2129, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2363 = stablehlo.transpose %2362, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %2364 = stablehlo.dot_general %2354, %2106#13, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %2365 = stablehlo.custom_call @Sharding(%2361) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2366 = stablehlo.custom_call @Sharding(%2364) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2367 = stablehlo.add %2365, %2366 : tensor<8x4096x576xbf16>
    %2368 = stablehlo.custom_call @Sharding(%2367) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2369 = stablehlo.multiply %2124, %2368 : tensor<8x4096x576xbf16>
    %cst_326 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2370 = stablehlo.reduce(%2369 init: %cst_326) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2371 = stablehlo.reshape %2370 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2372 = stablehlo.broadcast_in_dim %2125, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2373 = stablehlo.multiply %2368, %2372 : tensor<8x4096x576xbf16>
    %cst_327 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2374 = stablehlo.reduce(%2371 init: %cst_327) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2375 = stablehlo.convert %2373 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2376 = stablehlo.multiply %2108, %2375 : tensor<8x4096x576xf32>
    %cst_328 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2377 = stablehlo.reduce(%2376 init: %cst_328) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2378 = stablehlo.reshape %2377 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %2379 = stablehlo.broadcast_in_dim %2118, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2380 = stablehlo.multiply %2375, %2379 : tensor<8x4096x576xf32>
    %2381 = stablehlo.multiply %2378, %2121 : tensor<8x4096x1xf32>
    %cst_329 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2382 = stablehlo.broadcast_in_dim %cst_329, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2383 = stablehlo.divide %2381, %2382 : tensor<8x4096x1xf32>
    %cst_330 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2384 = stablehlo.reduce(%2383 init: %cst_330) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2385 = stablehlo.broadcast_in_dim %2384, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %2386 = stablehlo.multiply %2385, %2111 : tensor<8x4096x576xf32>
    %2387 = stablehlo.add %2380, %2386 : tensor<8x4096x576xf32>
    %2388 = stablehlo.convert %2387 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2389 = stablehlo.add %2281, %2388 : tensor<8x4096x576xbf16>
    %2390 = stablehlo.custom_call @Sharding(%2389) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2391:19 = stablehlo.optimization_barrier %426, %427, %428, %469, %489#1, %491, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %334, %arg203, %arg202, %2390 : tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096xi32>, tensor<8x4096xi32>, tensor<8x4096x576xbf16>
    %2392 = stablehlo.custom_call @Sharding(%2391#15) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2393 = stablehlo.convert %2392 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2394 = stablehlo.multiply %2393, %2393 : tensor<8x4096x576xf32>
    %cst_331 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %2395 = stablehlo.broadcast_in_dim %cst_331, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %2396 = stablehlo.multiply %2395, %2393 : tensor<8x4096x576xf32>
    %cst_332 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2397 = stablehlo.reduce(%2394 init: %cst_332) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2398 = stablehlo.broadcast_in_dim %2397, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_333 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2399 = stablehlo.broadcast_in_dim %cst_333, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2400 = stablehlo.divide %2398, %2399 : tensor<8x4096x1xf32>
    %cst_334 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2401 = stablehlo.broadcast_in_dim %cst_334, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2402 = stablehlo.add %2400, %2401 : tensor<8x4096x1xf32>
    %2403 = stablehlo.rsqrt %2402 : tensor<8x4096x1xf32>
    %2404 = stablehlo.divide %2403, %2402 : tensor<8x4096x1xf32>
    %cst_335 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %2405 = stablehlo.broadcast_in_dim %cst_335, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2406 = stablehlo.multiply %2405, %2404 : tensor<8x4096x1xf32>
    %2407 = stablehlo.broadcast_in_dim %2403, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2408 = stablehlo.multiply %2393, %2407 : tensor<8x4096x576xf32>
    %2409 = stablehlo.convert %2408 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2410 = stablehlo.broadcast_in_dim %2391#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2411 = stablehlo.broadcast_in_dim %2410, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2412 = stablehlo.multiply %2409, %2411 : tensor<8x4096x576xbf16>
    %2413 = stablehlo.custom_call @Sharding(%2412) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2414 = stablehlo.custom_call @Sharding(%2413) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2415 = stablehlo.custom_call @Sharding(%2413) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2416 = stablehlo.broadcast_in_dim %2391#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %2417 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_336 = stablehlo.constant dense<2> : tensor<i32>
    %2418 = stablehlo.broadcast_in_dim %c_336, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %2419 = stablehlo.multiply %2418, %2417 : tensor<64xi32>
    %2420 = stablehlo.convert %2419 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_337 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2421 = stablehlo.broadcast_in_dim %cst_337, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2422 = stablehlo.divide %2420, %2421 : tensor<64xf32>
    %cst_338 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %2423 = stablehlo.broadcast_in_dim %cst_338, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2424 = stablehlo.power %2423, %2422 : tensor<64xf32>
    %cst_339 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2425 = stablehlo.broadcast_in_dim %cst_339, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2426 = stablehlo.multiply %2425, %2424 : tensor<64xf32>
    %2427 = stablehlo.convert %2416 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %2428 = stablehlo.broadcast_in_dim %2426, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2429 = stablehlo.broadcast_in_dim %2427, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %2430 = stablehlo.broadcast_in_dim %2428, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %2431 = stablehlo.divide %2429, %2430 : tensor<8x4096x1x64xf32>
    %2432 = stablehlo.sine %2431 : tensor<8x4096x1x64xf32>
    %2433 = stablehlo.convert %2432 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %2434 = stablehlo.cosine %2431 : tensor<8x4096x1x64xf32>
    %2435 = stablehlo.convert %2434 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %2436 = stablehlo.broadcast_in_dim %2391#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %2437 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_340 = stablehlo.constant dense<2> : tensor<i32>
    %2438 = stablehlo.broadcast_in_dim %c_340, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %2439 = stablehlo.multiply %2438, %2437 : tensor<64xi32>
    %2440 = stablehlo.convert %2439 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_341 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2441 = stablehlo.broadcast_in_dim %cst_341, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2442 = stablehlo.divide %2440, %2441 : tensor<64xf32>
    %cst_342 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %2443 = stablehlo.broadcast_in_dim %cst_342, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2444 = stablehlo.power %2443, %2442 : tensor<64xf32>
    %cst_343 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2445 = stablehlo.broadcast_in_dim %cst_343, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2446 = stablehlo.multiply %2445, %2444 : tensor<64xf32>
    %2447 = stablehlo.convert %2436 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %2448 = stablehlo.broadcast_in_dim %2446, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2449 = stablehlo.broadcast_in_dim %2447, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %2450 = stablehlo.broadcast_in_dim %2448, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %2451 = stablehlo.divide %2449, %2450 : tensor<8x4096x1x64xf32>
    %2452 = stablehlo.sine %2451 : tensor<8x4096x1x64xf32>
    %2453 = stablehlo.convert %2452 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %2454 = stablehlo.cosine %2451 : tensor<8x4096x1x64xf32>
    %2455 = stablehlo.convert %2454 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %2456 = stablehlo.reshape %2391#0 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2457 = stablehlo.dot_general %2391#1, %2456, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %2458 = stablehlo.transpose %2457, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2459 = stablehlo.broadcast_in_dim %2391#16, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %2460 = stablehlo.broadcast_in_dim %2391#16, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %2461 = stablehlo.broadcast_in_dim %2459, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %2462 = stablehlo.broadcast_in_dim %2460, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %2463 = stablehlo.compare  EQ, %2461, %2462,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %2464 = stablehlo.broadcast_in_dim %2463, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %2465 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %2466 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_344 = stablehlo.constant dense<0> : tensor<i32>
    %2467 = stablehlo.broadcast_in_dim %c_344, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %2468 = stablehlo.add %2465, %2467 : tensor<4096x4096xi32>
    %2469 = stablehlo.compare  LE, %2466, %2468,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %2470 = stablehlo.broadcast_in_dim %2469, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %2471 = stablehlo.broadcast_in_dim %2470, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %2472 = stablehlo.and %2464, %2471 : tensor<8x1x1x4096x4096xi1>
    %cst_345 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_346 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %2473 = call @_where_115(%2472, %cst_345, %cst_346) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_347 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %2474 = stablehlo.broadcast_in_dim %cst_347, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %2475 = stablehlo.compare  GE, %2473, %2474,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_348 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %2476:2 = call @_where_116(%2475, %2458, %cst_348) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> (tensor<8x9x1x4096x4096xbf16>, tensor<8x9x1x4096x4096xi1>)
    %2477 = stablehlo.reshape %2476#0 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_349 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2478 = stablehlo.reduce(%2477 init: %cst_349) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2479 = stablehlo.reshape %2478 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %2480 = stablehlo.broadcast_in_dim %2479, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2481 = stablehlo.compare  EQ, %2477, %2480,  FLOAT : (tensor<8x9x4096x4096xbf16>, tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xi1>
    %2482 = stablehlo.convert %2481 : (tensor<8x9x4096x4096xi1>) -> tensor<8x9x4096x4096xbf16>
    %cst_350 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2483 = stablehlo.reduce(%2482 init: %cst_350) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2484 = stablehlo.broadcast_in_dim %2478, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %2485 = stablehlo.broadcast_in_dim %2484, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2486 = stablehlo.subtract %2477, %2485 : tensor<8x9x4096x4096xbf16>
    %2487 = stablehlo.exponential %2486 : tensor<8x9x4096x4096xbf16>
    %2488 = stablehlo.convert %2487 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_351 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2489 = stablehlo.reduce(%2488 init: %cst_351) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %2490 = stablehlo.broadcast_in_dim %2489, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %2491 = stablehlo.convert %2490 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %2492 = stablehlo.reshape %2487 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2493 = stablehlo.transpose %2491, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %2494 = stablehlo.dot_general %2391#2, %2492, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %2495 = stablehlo.transpose %2494, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2496 = stablehlo.reshape %2495 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2497 = stablehlo.broadcast_in_dim %2493, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %2498 = stablehlo.divide %2496, %2497 : tensor<8x4096x9x128xbf16>
    %2499 = stablehlo.multiply %2493, %2493 : tensor<8x4096x9x1xbf16>
    %cst_352 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2500 = stablehlo.broadcast_in_dim %cst_352, dims = [] : (tensor<bf16>) -> tensor<8x4096x9x1xbf16>
    %2501 = stablehlo.divide %2500, %2499 : tensor<8x4096x9x1xbf16>
    %2502 = stablehlo.custom_call @Sharding(%2498) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2503 = stablehlo.custom_call @Sharding(%2391#3) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2504 = stablehlo.add %2392, %2503 : tensor<8x4096x576xbf16>
    %2505 = stablehlo.convert %2504 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2506 = stablehlo.multiply %2505, %2505 : tensor<8x4096x576xf32>
    %cst_353 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %2507 = stablehlo.broadcast_in_dim %cst_353, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %2508 = stablehlo.multiply %2507, %2505 : tensor<8x4096x576xf32>
    %cst_354 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2509 = stablehlo.reduce(%2506 init: %cst_354) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2510 = stablehlo.broadcast_in_dim %2509, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_355 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2511 = stablehlo.broadcast_in_dim %cst_355, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2512 = stablehlo.divide %2510, %2511 : tensor<8x4096x1xf32>
    %cst_356 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2513 = stablehlo.broadcast_in_dim %cst_356, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2514 = stablehlo.add %2512, %2513 : tensor<8x4096x1xf32>
    %2515 = stablehlo.rsqrt %2514 : tensor<8x4096x1xf32>
    %2516 = stablehlo.divide %2515, %2514 : tensor<8x4096x1xf32>
    %cst_357 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %2517 = stablehlo.broadcast_in_dim %cst_357, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2518 = stablehlo.multiply %2517, %2516 : tensor<8x4096x1xf32>
    %2519 = stablehlo.broadcast_in_dim %2515, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2520 = stablehlo.multiply %2505, %2519 : tensor<8x4096x576xf32>
    %2521 = stablehlo.convert %2520 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2522 = stablehlo.broadcast_in_dim %2391#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2523 = stablehlo.broadcast_in_dim %2522, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2524 = stablehlo.multiply %2521, %2523 : tensor<8x4096x576xbf16>
    %2525 = stablehlo.custom_call @Sharding(%2524) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2526:3 = call @silu_124(%2391#4) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %2527 = stablehlo.multiply %2526#0, %2391#5 : tensor<8x4096x2016xbf16>
    %2528 = stablehlo.custom_call @Sharding(%2527) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %2529 = stablehlo.custom_call @Sharding(%2391#18) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2530 = stablehlo.custom_call @Sharding(%2529) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2531 = stablehlo.dot_general %2530, %2528, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2532 = stablehlo.transpose %2531, dims = [1, 0] : (tensor<576x2016xbf16>) -> tensor<2016x576xbf16>
    %2533 = stablehlo.dot_general %2530, %2391#8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x2016xbf16>
    %2534 = stablehlo.custom_call @Sharding(%2533) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %2535 = stablehlo.multiply %2526#0, %2534 : tensor<8x4096x2016xbf16>
    %2536 = stablehlo.multiply %2534, %2391#5 : tensor<8x4096x2016xbf16>
    %2537 = stablehlo.dot_general %2535, %2525, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %2538 = stablehlo.transpose %2537, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %2539 = stablehlo.dot_general %2535, %2391#7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %2540 = call @silu_133(%2526#1, %2391#4, %2526#2, %2536) : (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %2541 = stablehlo.dot_general %2540, %2525, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %2542 = stablehlo.transpose %2541, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %2543 = stablehlo.dot_general %2540, %2391#6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %2544 = stablehlo.add %2539, %2543 : tensor<8x4096x576xbf16>
    %2545 = stablehlo.custom_call @Sharding(%2544) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2546 = stablehlo.multiply %2521, %2545 : tensor<8x4096x576xbf16>
    %cst_358 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2547 = stablehlo.reduce(%2546 init: %cst_358) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2548 = stablehlo.reshape %2547 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2549 = stablehlo.broadcast_in_dim %2522, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2550 = stablehlo.multiply %2545, %2549 : tensor<8x4096x576xbf16>
    %cst_359 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2551 = stablehlo.reduce(%2548 init: %cst_359) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2552 = stablehlo.convert %2550 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2553 = stablehlo.multiply %2505, %2552 : tensor<8x4096x576xf32>
    %cst_360 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2554 = stablehlo.reduce(%2553 init: %cst_360) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2555 = stablehlo.reshape %2554 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %2556 = stablehlo.broadcast_in_dim %2515, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2557 = stablehlo.multiply %2552, %2556 : tensor<8x4096x576xf32>
    %2558 = stablehlo.multiply %2555, %2518 : tensor<8x4096x1xf32>
    %cst_361 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2559 = stablehlo.broadcast_in_dim %cst_361, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2560 = stablehlo.divide %2558, %2559 : tensor<8x4096x1xf32>
    %cst_362 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2561 = stablehlo.reduce(%2560 init: %cst_362) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2562 = stablehlo.broadcast_in_dim %2561, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %2563 = stablehlo.multiply %2562, %2508 : tensor<8x4096x576xf32>
    %2564 = stablehlo.add %2557, %2563 : tensor<8x4096x576xf32>
    %2565 = stablehlo.convert %2564 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2566 = stablehlo.add %2529, %2565 : tensor<8x4096x576xbf16>
    %2567 = stablehlo.custom_call @Sharding(%2566) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2568 = stablehlo.dot_general %2567, %2502, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2569 = stablehlo.transpose %2568, dims = [1, 2, 0] : (tensor<576x9x128xbf16>) -> tensor<9x128x576xbf16>
    %2570 = stablehlo.dot_general %2567, %2391#12, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x9x128xbf16>
    %2571 = stablehlo.custom_call @Sharding(%2570) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2572 = stablehlo.broadcast_in_dim %2501, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %2573 = stablehlo.multiply %2571, %2572 : tensor<8x4096x9x128xbf16>
    %2574 = stablehlo.multiply %2573, %2496 : tensor<8x4096x9x128xbf16>
    %cst_363 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2575 = stablehlo.reduce(%2574 init: %cst_363) applies stablehlo.add across dimensions = [3] : (tensor<8x4096x9x128xbf16>, tensor<bf16>) -> tensor<8x4096x9xbf16>
    %2576 = stablehlo.reshape %2575 : (tensor<8x4096x9xbf16>) -> tensor<8x4096x9x1xbf16>
    %2577 = stablehlo.negate %2576 : tensor<8x4096x9x1xbf16>
    %2578 = stablehlo.broadcast_in_dim %2493, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %2579 = stablehlo.divide %2571, %2578 : tensor<8x4096x9x128xbf16>
    %2580 = stablehlo.reshape %2579 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2581 = stablehlo.transpose %2580, dims = [0, 2, 4, 3, 1] : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %2582 = stablehlo.dot_general %2581, %2391#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2583 = stablehlo.dot_general %2581, %2492, batching_dims = [0, 1] x [0, 1], contracting_dims = [3, 4] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x4096xbf16>
    %2584 = stablehlo.transpose %2583, dims = [0, 3, 1, 2] : (tensor<8x9x128x4096xbf16>) -> tensor<8x4096x9x128xbf16>
    %2585 = stablehlo.transpose %2577, dims = [0, 2, 1, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x9x4096x1xbf16>
    %2586 = stablehlo.reshape %2582 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2587 = stablehlo.convert %2585 : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x1xf32>
    %cst_364 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2588 = stablehlo.reduce(%2587 init: %cst_364) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %2589 = stablehlo.broadcast_in_dim %2588, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x4096xf32>
    %2590 = stablehlo.convert %2589 : (tensor<8x9x4096x4096xf32>) -> tensor<8x9x4096x4096xbf16>
    %2591 = stablehlo.add %2586, %2590 : tensor<8x9x4096x4096xbf16>
    %2592 = stablehlo.multiply %2591, %2487 : tensor<8x9x4096x4096xbf16>
    %2593 = stablehlo.negate %2592 : tensor<8x9x4096x4096xbf16>
    %cst_365 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2594 = stablehlo.reduce(%2593 init: %cst_365) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2595 = stablehlo.reshape %2594 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %cst_366 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2596 = stablehlo.reduce(%2595 init: %cst_366) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2597 = stablehlo.divide %2596, %2483 : tensor<8x9x4096xbf16>
    %2598 = stablehlo.broadcast_in_dim %2597, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2599 = stablehlo.multiply %2598, %2482 : tensor<8x9x4096x4096xbf16>
    %2600 = stablehlo.add %2592, %2599 : tensor<8x9x4096x4096xbf16>
    %2601 = stablehlo.reshape %2600 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2602 = call @_where_163(%2476#1, %2601) : (tensor<8x9x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2603 = stablehlo.transpose %2602, dims = [0, 1, 4, 3, 2] : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %2604 = stablehlo.dot_general %2603, %2391#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x4096x1x128xbf16>
    %2605 = stablehlo.transpose %2604, dims = [0, 2, 1, 3, 4] : (tensor<8x9x4096x1x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2606 = stablehlo.dot_general %2603, %2456, batching_dims = [0, 1] x [0, 2], contracting_dims = [3, 4] x [1, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x128xbf16>
    %2607 = stablehlo.transpose %2606, dims = [0, 2, 1, 3] : (tensor<8x9x4096x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2608 = stablehlo.reshape %2605 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2609 = stablehlo.custom_call @Sharding(%2584) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2610 = stablehlo.custom_call @Sharding(%2607) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2611 = stablehlo.custom_call @Sharding(%2608) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2612 = stablehlo.slice %2610 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2613 = stablehlo.slice %2610 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2614 = stablehlo.broadcast_in_dim %2453, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2615 = stablehlo.multiply %2613, %2614 : tensor<8x4096x9x64xbf16>
    %2616 = stablehlo.broadcast_in_dim %2455, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2617 = stablehlo.multiply %2613, %2616 : tensor<8x4096x9x64xbf16>
    %2618 = stablehlo.negate %2612 : tensor<8x4096x9x64xbf16>
    %2619 = stablehlo.broadcast_in_dim %2453, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2620 = stablehlo.multiply %2618, %2619 : tensor<8x4096x9x64xbf16>
    %2621 = stablehlo.add %2617, %2620 : tensor<8x4096x9x64xbf16>
    %2622 = stablehlo.broadcast_in_dim %2455, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2623 = stablehlo.multiply %2612, %2622 : tensor<8x4096x9x64xbf16>
    %2624 = stablehlo.add %2615, %2623 : tensor<8x4096x9x64xbf16>
    %2625 = stablehlo.concatenate %2624, %2621, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %2626 = stablehlo.slice %2611 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2627 = stablehlo.slice %2611 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2628 = stablehlo.broadcast_in_dim %2433, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2629 = stablehlo.multiply %2627, %2628 : tensor<8x4096x9x64xbf16>
    %2630 = stablehlo.broadcast_in_dim %2435, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2631 = stablehlo.multiply %2627, %2630 : tensor<8x4096x9x64xbf16>
    %2632 = stablehlo.negate %2626 : tensor<8x4096x9x64xbf16>
    %2633 = stablehlo.broadcast_in_dim %2433, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2634 = stablehlo.multiply %2632, %2633 : tensor<8x4096x9x64xbf16>
    %2635 = stablehlo.add %2631, %2634 : tensor<8x4096x9x64xbf16>
    %2636 = stablehlo.broadcast_in_dim %2435, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2637 = stablehlo.multiply %2626, %2636 : tensor<8x4096x9x64xbf16>
    %2638 = stablehlo.add %2629, %2637 : tensor<8x4096x9x64xbf16>
    %2639 = stablehlo.concatenate %2638, %2635, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %2640 = stablehlo.dot_general %2609, %2415, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2641 = stablehlo.transpose %2640, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %2642 = stablehlo.dot_general %2609, %2391#14, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %2643 = stablehlo.dot_general %2625, %2415, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2644 = stablehlo.transpose %2643, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %2645 = stablehlo.dot_general %2625, %2391#11, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %2646 = stablehlo.add %2642, %2645 : tensor<8x4096x576xbf16>
    %2647 = stablehlo.dot_general %2639, %2414, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2648 = stablehlo.transpose %2647, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %2649 = stablehlo.dot_general %2639, %2391#13, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %2650 = stablehlo.custom_call @Sharding(%2646) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2651 = stablehlo.custom_call @Sharding(%2649) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2652 = stablehlo.add %2650, %2651 : tensor<8x4096x576xbf16>
    %2653 = stablehlo.custom_call @Sharding(%2652) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2654 = stablehlo.multiply %2409, %2653 : tensor<8x4096x576xbf16>
    %cst_367 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2655 = stablehlo.reduce(%2654 init: %cst_367) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2656 = stablehlo.reshape %2655 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2657 = stablehlo.broadcast_in_dim %2410, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2658 = stablehlo.multiply %2653, %2657 : tensor<8x4096x576xbf16>
    %cst_368 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2659 = stablehlo.reduce(%2656 init: %cst_368) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2660 = stablehlo.convert %2658 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2661 = stablehlo.multiply %2393, %2660 : tensor<8x4096x576xf32>
    %cst_369 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2662 = stablehlo.reduce(%2661 init: %cst_369) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2663 = stablehlo.reshape %2662 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %2664 = stablehlo.broadcast_in_dim %2403, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2665 = stablehlo.multiply %2660, %2664 : tensor<8x4096x576xf32>
    %2666 = stablehlo.multiply %2663, %2406 : tensor<8x4096x1xf32>
    %cst_370 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2667 = stablehlo.broadcast_in_dim %cst_370, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2668 = stablehlo.divide %2666, %2667 : tensor<8x4096x1xf32>
    %cst_371 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2669 = stablehlo.reduce(%2668 init: %cst_371) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2670 = stablehlo.broadcast_in_dim %2669, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %2671 = stablehlo.multiply %2670, %2396 : tensor<8x4096x576xf32>
    %2672 = stablehlo.add %2665, %2671 : tensor<8x4096x576xf32>
    %2673 = stablehlo.convert %2672 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2674 = stablehlo.add %2566, %2673 : tensor<8x4096x576xbf16>
    %2675 = stablehlo.custom_call @Sharding(%2674) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2676:19 = stablehlo.optimization_barrier %263, %264, %265, %306, %326#1, %328, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %171, %arg203, %arg202, %2675 : tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096xi32>, tensor<8x4096xi32>, tensor<8x4096x576xbf16>
    %2677 = stablehlo.custom_call @Sharding(%2676#15) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2678 = stablehlo.convert %2677 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2679 = stablehlo.multiply %2678, %2678 : tensor<8x4096x576xf32>
    %cst_372 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %2680 = stablehlo.broadcast_in_dim %cst_372, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %2681 = stablehlo.multiply %2680, %2678 : tensor<8x4096x576xf32>
    %cst_373 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2682 = stablehlo.reduce(%2679 init: %cst_373) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2683 = stablehlo.broadcast_in_dim %2682, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_374 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2684 = stablehlo.broadcast_in_dim %cst_374, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2685 = stablehlo.divide %2683, %2684 : tensor<8x4096x1xf32>
    %cst_375 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2686 = stablehlo.broadcast_in_dim %cst_375, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2687 = stablehlo.add %2685, %2686 : tensor<8x4096x1xf32>
    %2688 = stablehlo.rsqrt %2687 : tensor<8x4096x1xf32>
    %2689 = stablehlo.divide %2688, %2687 : tensor<8x4096x1xf32>
    %cst_376 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %2690 = stablehlo.broadcast_in_dim %cst_376, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2691 = stablehlo.multiply %2690, %2689 : tensor<8x4096x1xf32>
    %2692 = stablehlo.broadcast_in_dim %2688, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2693 = stablehlo.multiply %2678, %2692 : tensor<8x4096x576xf32>
    %2694 = stablehlo.convert %2693 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2695 = stablehlo.broadcast_in_dim %2676#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2696 = stablehlo.broadcast_in_dim %2695, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2697 = stablehlo.multiply %2694, %2696 : tensor<8x4096x576xbf16>
    %2698 = stablehlo.custom_call @Sharding(%2697) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2699 = stablehlo.custom_call @Sharding(%2698) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2700 = stablehlo.custom_call @Sharding(%2698) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2701 = stablehlo.broadcast_in_dim %2676#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %2702 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_377 = stablehlo.constant dense<2> : tensor<i32>
    %2703 = stablehlo.broadcast_in_dim %c_377, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %2704 = stablehlo.multiply %2703, %2702 : tensor<64xi32>
    %2705 = stablehlo.convert %2704 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_378 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2706 = stablehlo.broadcast_in_dim %cst_378, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2707 = stablehlo.divide %2705, %2706 : tensor<64xf32>
    %cst_379 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %2708 = stablehlo.broadcast_in_dim %cst_379, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2709 = stablehlo.power %2708, %2707 : tensor<64xf32>
    %cst_380 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2710 = stablehlo.broadcast_in_dim %cst_380, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2711 = stablehlo.multiply %2710, %2709 : tensor<64xf32>
    %2712 = stablehlo.convert %2701 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %2713 = stablehlo.broadcast_in_dim %2711, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2714 = stablehlo.broadcast_in_dim %2712, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %2715 = stablehlo.broadcast_in_dim %2713, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %2716 = stablehlo.divide %2714, %2715 : tensor<8x4096x1x64xf32>
    %2717 = stablehlo.sine %2716 : tensor<8x4096x1x64xf32>
    %2718 = stablehlo.convert %2717 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %2719 = stablehlo.cosine %2716 : tensor<8x4096x1x64xf32>
    %2720 = stablehlo.convert %2719 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %2721 = stablehlo.broadcast_in_dim %2676#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %2722 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_381 = stablehlo.constant dense<2> : tensor<i32>
    %2723 = stablehlo.broadcast_in_dim %c_381, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %2724 = stablehlo.multiply %2723, %2722 : tensor<64xi32>
    %2725 = stablehlo.convert %2724 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_382 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2726 = stablehlo.broadcast_in_dim %cst_382, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2727 = stablehlo.divide %2725, %2726 : tensor<64xf32>
    %cst_383 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %2728 = stablehlo.broadcast_in_dim %cst_383, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2729 = stablehlo.power %2728, %2727 : tensor<64xf32>
    %cst_384 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2730 = stablehlo.broadcast_in_dim %cst_384, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2731 = stablehlo.multiply %2730, %2729 : tensor<64xf32>
    %2732 = stablehlo.convert %2721 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %2733 = stablehlo.broadcast_in_dim %2731, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2734 = stablehlo.broadcast_in_dim %2732, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %2735 = stablehlo.broadcast_in_dim %2733, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %2736 = stablehlo.divide %2734, %2735 : tensor<8x4096x1x64xf32>
    %2737 = stablehlo.sine %2736 : tensor<8x4096x1x64xf32>
    %2738 = stablehlo.convert %2737 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %2739 = stablehlo.cosine %2736 : tensor<8x4096x1x64xf32>
    %2740 = stablehlo.convert %2739 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %2741 = stablehlo.reshape %2676#0 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2742 = stablehlo.dot_general %2676#1, %2741, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %2743 = stablehlo.transpose %2742, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2744 = stablehlo.broadcast_in_dim %2676#16, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %2745 = stablehlo.broadcast_in_dim %2676#16, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %2746 = stablehlo.broadcast_in_dim %2744, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %2747 = stablehlo.broadcast_in_dim %2745, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %2748 = stablehlo.compare  EQ, %2746, %2747,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %2749 = stablehlo.broadcast_in_dim %2748, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %2750 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %2751 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_385 = stablehlo.constant dense<0> : tensor<i32>
    %2752 = stablehlo.broadcast_in_dim %c_385, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %2753 = stablehlo.add %2750, %2752 : tensor<4096x4096xi32>
    %2754 = stablehlo.compare  LE, %2751, %2753,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %2755 = stablehlo.broadcast_in_dim %2754, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %2756 = stablehlo.broadcast_in_dim %2755, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %2757 = stablehlo.and %2749, %2756 : tensor<8x1x1x4096x4096xi1>
    %cst_386 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_387 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %2758 = call @_where_115(%2757, %cst_386, %cst_387) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_388 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %2759 = stablehlo.broadcast_in_dim %cst_388, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %2760 = stablehlo.compare  GE, %2758, %2759,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_389 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %2761:2 = call @_where_116(%2760, %2743, %cst_389) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> (tensor<8x9x1x4096x4096xbf16>, tensor<8x9x1x4096x4096xi1>)
    %2762 = stablehlo.reshape %2761#0 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_390 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2763 = stablehlo.reduce(%2762 init: %cst_390) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2764 = stablehlo.reshape %2763 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %2765 = stablehlo.broadcast_in_dim %2764, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2766 = stablehlo.compare  EQ, %2762, %2765,  FLOAT : (tensor<8x9x4096x4096xbf16>, tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xi1>
    %2767 = stablehlo.convert %2766 : (tensor<8x9x4096x4096xi1>) -> tensor<8x9x4096x4096xbf16>
    %cst_391 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2768 = stablehlo.reduce(%2767 init: %cst_391) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2769 = stablehlo.broadcast_in_dim %2763, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %2770 = stablehlo.broadcast_in_dim %2769, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2771 = stablehlo.subtract %2762, %2770 : tensor<8x9x4096x4096xbf16>
    %2772 = stablehlo.exponential %2771 : tensor<8x9x4096x4096xbf16>
    %2773 = stablehlo.convert %2772 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_392 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2774 = stablehlo.reduce(%2773 init: %cst_392) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %2775 = stablehlo.broadcast_in_dim %2774, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %2776 = stablehlo.convert %2775 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %2777 = stablehlo.reshape %2772 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2778 = stablehlo.transpose %2776, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %2779 = stablehlo.dot_general %2676#2, %2777, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %2780 = stablehlo.transpose %2779, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2781 = stablehlo.reshape %2780 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2782 = stablehlo.broadcast_in_dim %2778, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %2783 = stablehlo.divide %2781, %2782 : tensor<8x4096x9x128xbf16>
    %2784 = stablehlo.multiply %2778, %2778 : tensor<8x4096x9x1xbf16>
    %cst_393 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2785 = stablehlo.broadcast_in_dim %cst_393, dims = [] : (tensor<bf16>) -> tensor<8x4096x9x1xbf16>
    %2786 = stablehlo.divide %2785, %2784 : tensor<8x4096x9x1xbf16>
    %2787 = stablehlo.custom_call @Sharding(%2783) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2788 = stablehlo.custom_call @Sharding(%2676#3) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2789 = stablehlo.add %2677, %2788 : tensor<8x4096x576xbf16>
    %2790 = stablehlo.convert %2789 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2791 = stablehlo.multiply %2790, %2790 : tensor<8x4096x576xf32>
    %cst_394 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %2792 = stablehlo.broadcast_in_dim %cst_394, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %2793 = stablehlo.multiply %2792, %2790 : tensor<8x4096x576xf32>
    %cst_395 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2794 = stablehlo.reduce(%2791 init: %cst_395) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2795 = stablehlo.broadcast_in_dim %2794, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_396 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2796 = stablehlo.broadcast_in_dim %cst_396, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2797 = stablehlo.divide %2795, %2796 : tensor<8x4096x1xf32>
    %cst_397 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2798 = stablehlo.broadcast_in_dim %cst_397, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2799 = stablehlo.add %2797, %2798 : tensor<8x4096x1xf32>
    %2800 = stablehlo.rsqrt %2799 : tensor<8x4096x1xf32>
    %2801 = stablehlo.divide %2800, %2799 : tensor<8x4096x1xf32>
    %cst_398 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %2802 = stablehlo.broadcast_in_dim %cst_398, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2803 = stablehlo.multiply %2802, %2801 : tensor<8x4096x1xf32>
    %2804 = stablehlo.broadcast_in_dim %2800, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2805 = stablehlo.multiply %2790, %2804 : tensor<8x4096x576xf32>
    %2806 = stablehlo.convert %2805 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2807 = stablehlo.broadcast_in_dim %2676#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2808 = stablehlo.broadcast_in_dim %2807, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2809 = stablehlo.multiply %2806, %2808 : tensor<8x4096x576xbf16>
    %2810 = stablehlo.custom_call @Sharding(%2809) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2811:3 = call @silu_124(%2676#4) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %2812 = stablehlo.multiply %2811#0, %2676#5 : tensor<8x4096x2016xbf16>
    %2813 = stablehlo.custom_call @Sharding(%2812) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %2814 = stablehlo.custom_call @Sharding(%2676#18) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2815 = stablehlo.custom_call @Sharding(%2814) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2816 = stablehlo.dot_general %2815, %2813, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2817 = stablehlo.transpose %2816, dims = [1, 0] : (tensor<576x2016xbf16>) -> tensor<2016x576xbf16>
    %2818 = stablehlo.dot_general %2815, %2676#8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x2016xbf16>
    %2819 = stablehlo.custom_call @Sharding(%2818) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %2820 = stablehlo.multiply %2811#0, %2819 : tensor<8x4096x2016xbf16>
    %2821 = stablehlo.multiply %2819, %2676#5 : tensor<8x4096x2016xbf16>
    %2822 = stablehlo.dot_general %2820, %2810, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %2823 = stablehlo.transpose %2822, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %2824 = stablehlo.dot_general %2820, %2676#7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %2825 = call @silu_133(%2811#1, %2676#4, %2811#2, %2821) : (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %2826 = stablehlo.dot_general %2825, %2810, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %2827 = stablehlo.transpose %2826, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %2828 = stablehlo.dot_general %2825, %2676#6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %2829 = stablehlo.add %2824, %2828 : tensor<8x4096x576xbf16>
    %2830 = stablehlo.custom_call @Sharding(%2829) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2831 = stablehlo.multiply %2806, %2830 : tensor<8x4096x576xbf16>
    %cst_399 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2832 = stablehlo.reduce(%2831 init: %cst_399) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2833 = stablehlo.reshape %2832 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2834 = stablehlo.broadcast_in_dim %2807, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2835 = stablehlo.multiply %2830, %2834 : tensor<8x4096x576xbf16>
    %cst_400 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2836 = stablehlo.reduce(%2833 init: %cst_400) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2837 = stablehlo.convert %2835 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2838 = stablehlo.multiply %2790, %2837 : tensor<8x4096x576xf32>
    %cst_401 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2839 = stablehlo.reduce(%2838 init: %cst_401) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2840 = stablehlo.reshape %2839 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %2841 = stablehlo.broadcast_in_dim %2800, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2842 = stablehlo.multiply %2837, %2841 : tensor<8x4096x576xf32>
    %2843 = stablehlo.multiply %2840, %2803 : tensor<8x4096x1xf32>
    %cst_402 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2844 = stablehlo.broadcast_in_dim %cst_402, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2845 = stablehlo.divide %2843, %2844 : tensor<8x4096x1xf32>
    %cst_403 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2846 = stablehlo.reduce(%2845 init: %cst_403) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2847 = stablehlo.broadcast_in_dim %2846, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %2848 = stablehlo.multiply %2847, %2793 : tensor<8x4096x576xf32>
    %2849 = stablehlo.add %2842, %2848 : tensor<8x4096x576xf32>
    %2850 = stablehlo.convert %2849 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2851 = stablehlo.add %2814, %2850 : tensor<8x4096x576xbf16>
    %2852 = stablehlo.custom_call @Sharding(%2851) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2853 = stablehlo.dot_general %2852, %2787, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2854 = stablehlo.transpose %2853, dims = [1, 2, 0] : (tensor<576x9x128xbf16>) -> tensor<9x128x576xbf16>
    %2855 = stablehlo.dot_general %2852, %2676#12, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x9x128xbf16>
    %2856 = stablehlo.custom_call @Sharding(%2855) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2857 = stablehlo.broadcast_in_dim %2786, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %2858 = stablehlo.multiply %2856, %2857 : tensor<8x4096x9x128xbf16>
    %2859 = stablehlo.multiply %2858, %2781 : tensor<8x4096x9x128xbf16>
    %cst_404 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2860 = stablehlo.reduce(%2859 init: %cst_404) applies stablehlo.add across dimensions = [3] : (tensor<8x4096x9x128xbf16>, tensor<bf16>) -> tensor<8x4096x9xbf16>
    %2861 = stablehlo.reshape %2860 : (tensor<8x4096x9xbf16>) -> tensor<8x4096x9x1xbf16>
    %2862 = stablehlo.negate %2861 : tensor<8x4096x9x1xbf16>
    %2863 = stablehlo.broadcast_in_dim %2778, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %2864 = stablehlo.divide %2856, %2863 : tensor<8x4096x9x128xbf16>
    %2865 = stablehlo.reshape %2864 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2866 = stablehlo.transpose %2865, dims = [0, 2, 4, 3, 1] : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %2867 = stablehlo.dot_general %2866, %2676#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2868 = stablehlo.dot_general %2866, %2777, batching_dims = [0, 1] x [0, 1], contracting_dims = [3, 4] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x4096xbf16>
    %2869 = stablehlo.transpose %2868, dims = [0, 3, 1, 2] : (tensor<8x9x128x4096xbf16>) -> tensor<8x4096x9x128xbf16>
    %2870 = stablehlo.transpose %2862, dims = [0, 2, 1, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x9x4096x1xbf16>
    %2871 = stablehlo.reshape %2867 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2872 = stablehlo.convert %2870 : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x1xf32>
    %cst_405 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2873 = stablehlo.reduce(%2872 init: %cst_405) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %2874 = stablehlo.broadcast_in_dim %2873, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x4096xf32>
    %2875 = stablehlo.convert %2874 : (tensor<8x9x4096x4096xf32>) -> tensor<8x9x4096x4096xbf16>
    %2876 = stablehlo.add %2871, %2875 : tensor<8x9x4096x4096xbf16>
    %2877 = stablehlo.multiply %2876, %2772 : tensor<8x9x4096x4096xbf16>
    %2878 = stablehlo.negate %2877 : tensor<8x9x4096x4096xbf16>
    %cst_406 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2879 = stablehlo.reduce(%2878 init: %cst_406) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2880 = stablehlo.reshape %2879 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %cst_407 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2881 = stablehlo.reduce(%2880 init: %cst_407) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %2882 = stablehlo.divide %2881, %2768 : tensor<8x9x4096xbf16>
    %2883 = stablehlo.broadcast_in_dim %2882, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %2884 = stablehlo.multiply %2883, %2767 : tensor<8x9x4096x4096xbf16>
    %2885 = stablehlo.add %2877, %2884 : tensor<8x9x4096x4096xbf16>
    %2886 = stablehlo.reshape %2885 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2887 = call @_where_163(%2761#1, %2886) : (tensor<8x9x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %2888 = stablehlo.transpose %2887, dims = [0, 1, 4, 3, 2] : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %2889 = stablehlo.dot_general %2888, %2676#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x4096x1x128xbf16>
    %2890 = stablehlo.transpose %2889, dims = [0, 2, 1, 3, 4] : (tensor<8x9x4096x1x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %2891 = stablehlo.dot_general %2888, %2741, batching_dims = [0, 1] x [0, 2], contracting_dims = [3, 4] x [1, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x128xbf16>
    %2892 = stablehlo.transpose %2891, dims = [0, 2, 1, 3] : (tensor<8x9x4096x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2893 = stablehlo.reshape %2890 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2894 = stablehlo.custom_call @Sharding(%2869) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2895 = stablehlo.custom_call @Sharding(%2892) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2896 = stablehlo.custom_call @Sharding(%2893) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %2897 = stablehlo.slice %2895 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2898 = stablehlo.slice %2895 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2899 = stablehlo.broadcast_in_dim %2738, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2900 = stablehlo.multiply %2898, %2899 : tensor<8x4096x9x64xbf16>
    %2901 = stablehlo.broadcast_in_dim %2740, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2902 = stablehlo.multiply %2898, %2901 : tensor<8x4096x9x64xbf16>
    %2903 = stablehlo.negate %2897 : tensor<8x4096x9x64xbf16>
    %2904 = stablehlo.broadcast_in_dim %2738, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2905 = stablehlo.multiply %2903, %2904 : tensor<8x4096x9x64xbf16>
    %2906 = stablehlo.add %2902, %2905 : tensor<8x4096x9x64xbf16>
    %2907 = stablehlo.broadcast_in_dim %2740, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2908 = stablehlo.multiply %2897, %2907 : tensor<8x4096x9x64xbf16>
    %2909 = stablehlo.add %2900, %2908 : tensor<8x4096x9x64xbf16>
    %2910 = stablehlo.concatenate %2909, %2906, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %2911 = stablehlo.slice %2896 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2912 = stablehlo.slice %2896 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %2913 = stablehlo.broadcast_in_dim %2718, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2914 = stablehlo.multiply %2912, %2913 : tensor<8x4096x9x64xbf16>
    %2915 = stablehlo.broadcast_in_dim %2720, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2916 = stablehlo.multiply %2912, %2915 : tensor<8x4096x9x64xbf16>
    %2917 = stablehlo.negate %2911 : tensor<8x4096x9x64xbf16>
    %2918 = stablehlo.broadcast_in_dim %2718, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2919 = stablehlo.multiply %2917, %2918 : tensor<8x4096x9x64xbf16>
    %2920 = stablehlo.add %2916, %2919 : tensor<8x4096x9x64xbf16>
    %2921 = stablehlo.broadcast_in_dim %2720, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %2922 = stablehlo.multiply %2911, %2921 : tensor<8x4096x9x64xbf16>
    %2923 = stablehlo.add %2914, %2922 : tensor<8x4096x9x64xbf16>
    %2924 = stablehlo.concatenate %2923, %2920, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %2925 = stablehlo.dot_general %2894, %2700, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2926 = stablehlo.transpose %2925, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %2927 = stablehlo.dot_general %2894, %2676#14, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %2928 = stablehlo.dot_general %2910, %2700, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2929 = stablehlo.transpose %2928, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %2930 = stablehlo.dot_general %2910, %2676#11, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %2931 = stablehlo.add %2927, %2930 : tensor<8x4096x576xbf16>
    %2932 = stablehlo.dot_general %2924, %2699, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2933 = stablehlo.transpose %2932, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %2934 = stablehlo.dot_general %2924, %2676#13, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %2935 = stablehlo.custom_call @Sharding(%2931) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2936 = stablehlo.custom_call @Sharding(%2934) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2937 = stablehlo.add %2935, %2936 : tensor<8x4096x576xbf16>
    %2938 = stablehlo.custom_call @Sharding(%2937) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2939 = stablehlo.multiply %2694, %2938 : tensor<8x4096x576xbf16>
    %cst_408 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2940 = stablehlo.reduce(%2939 init: %cst_408) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2941 = stablehlo.reshape %2940 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2942 = stablehlo.broadcast_in_dim %2695, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2943 = stablehlo.multiply %2938, %2942 : tensor<8x4096x576xbf16>
    %cst_409 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2944 = stablehlo.reduce(%2941 init: %cst_409) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2945 = stablehlo.convert %2943 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2946 = stablehlo.multiply %2678, %2945 : tensor<8x4096x576xf32>
    %cst_410 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2947 = stablehlo.reduce(%2946 init: %cst_410) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2948 = stablehlo.reshape %2947 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %2949 = stablehlo.broadcast_in_dim %2688, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2950 = stablehlo.multiply %2945, %2949 : tensor<8x4096x576xf32>
    %2951 = stablehlo.multiply %2948, %2691 : tensor<8x4096x1xf32>
    %cst_411 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2952 = stablehlo.broadcast_in_dim %cst_411, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2953 = stablehlo.divide %2951, %2952 : tensor<8x4096x1xf32>
    %cst_412 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2954 = stablehlo.reduce(%2953 init: %cst_412) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2955 = stablehlo.broadcast_in_dim %2954, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %2956 = stablehlo.multiply %2955, %2681 : tensor<8x4096x576xf32>
    %2957 = stablehlo.add %2950, %2956 : tensor<8x4096x576xf32>
    %2958 = stablehlo.convert %2957 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2959 = stablehlo.add %2851, %2958 : tensor<8x4096x576xbf16>
    %2960 = stablehlo.custom_call @Sharding(%2959) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2961:19 = stablehlo.optimization_barrier %100, %101, %102, %143, %163#1, %165, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %8, %arg203, %arg202, %2960 : tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<8x4096x576xbf16>, tensor<8x4096xi32>, tensor<8x4096xi32>, tensor<8x4096x576xbf16>
    %2962 = stablehlo.custom_call @Sharding(%2961#15) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2963 = stablehlo.convert %2962 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %2964 = stablehlo.multiply %2963, %2963 : tensor<8x4096x576xf32>
    %cst_413 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %2965 = stablehlo.broadcast_in_dim %cst_413, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %2966 = stablehlo.multiply %2965, %2963 : tensor<8x4096x576xf32>
    %cst_414 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2967 = stablehlo.reduce(%2964 init: %cst_414) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %2968 = stablehlo.broadcast_in_dim %2967, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_415 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %2969 = stablehlo.broadcast_in_dim %cst_415, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2970 = stablehlo.divide %2968, %2969 : tensor<8x4096x1xf32>
    %cst_416 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2971 = stablehlo.broadcast_in_dim %cst_416, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2972 = stablehlo.add %2970, %2971 : tensor<8x4096x1xf32>
    %2973 = stablehlo.rsqrt %2972 : tensor<8x4096x1xf32>
    %2974 = stablehlo.divide %2973, %2972 : tensor<8x4096x1xf32>
    %cst_417 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %2975 = stablehlo.broadcast_in_dim %cst_417, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %2976 = stablehlo.multiply %2975, %2974 : tensor<8x4096x1xf32>
    %2977 = stablehlo.broadcast_in_dim %2973, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %2978 = stablehlo.multiply %2963, %2977 : tensor<8x4096x576xf32>
    %2979 = stablehlo.convert %2978 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %2980 = stablehlo.broadcast_in_dim %2961#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %2981 = stablehlo.broadcast_in_dim %2980, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %2982 = stablehlo.multiply %2979, %2981 : tensor<8x4096x576xbf16>
    %2983 = stablehlo.custom_call @Sharding(%2982) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2984 = stablehlo.custom_call @Sharding(%2983) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2985 = stablehlo.custom_call @Sharding(%2983) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %2986 = stablehlo.broadcast_in_dim %2961#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %2987 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_418 = stablehlo.constant dense<2> : tensor<i32>
    %2988 = stablehlo.broadcast_in_dim %c_418, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %2989 = stablehlo.multiply %2988, %2987 : tensor<64xi32>
    %2990 = stablehlo.convert %2989 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_419 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2991 = stablehlo.broadcast_in_dim %cst_419, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2992 = stablehlo.divide %2990, %2991 : tensor<64xf32>
    %cst_420 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %2993 = stablehlo.broadcast_in_dim %cst_420, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2994 = stablehlo.power %2993, %2992 : tensor<64xf32>
    %cst_421 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2995 = stablehlo.broadcast_in_dim %cst_421, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2996 = stablehlo.multiply %2995, %2994 : tensor<64xf32>
    %2997 = stablehlo.convert %2986 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %2998 = stablehlo.broadcast_in_dim %2996, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2999 = stablehlo.broadcast_in_dim %2997, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %3000 = stablehlo.broadcast_in_dim %2998, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %3001 = stablehlo.divide %2999, %3000 : tensor<8x4096x1x64xf32>
    %3002 = stablehlo.sine %3001 : tensor<8x4096x1x64xf32>
    %3003 = stablehlo.convert %3002 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %3004 = stablehlo.cosine %3001 : tensor<8x4096x1x64xf32>
    %3005 = stablehlo.convert %3004 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %3006 = stablehlo.broadcast_in_dim %2961#17, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1x1xi32>
    %3007 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_422 = stablehlo.constant dense<2> : tensor<i32>
    %3008 = stablehlo.broadcast_in_dim %c_422, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %3009 = stablehlo.multiply %3008, %3007 : tensor<64xi32>
    %3010 = stablehlo.convert %3009 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_423 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %3011 = stablehlo.broadcast_in_dim %cst_423, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3012 = stablehlo.divide %3010, %3011 : tensor<64xf32>
    %cst_424 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %3013 = stablehlo.broadcast_in_dim %cst_424, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3014 = stablehlo.power %3013, %3012 : tensor<64xf32>
    %cst_425 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3015 = stablehlo.broadcast_in_dim %cst_425, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3016 = stablehlo.multiply %3015, %3014 : tensor<64xf32>
    %3017 = stablehlo.convert %3006 : (tensor<8x4096x1x1xi32>) -> tensor<8x4096x1x1xf32>
    %3018 = stablehlo.broadcast_in_dim %3016, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %3019 = stablehlo.broadcast_in_dim %3017, dims = [0, 1, 2, 3] : (tensor<8x4096x1x1xf32>) -> tensor<8x4096x1x64xf32>
    %3020 = stablehlo.broadcast_in_dim %3018, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<8x4096x1x64xf32>
    %3021 = stablehlo.divide %3019, %3020 : tensor<8x4096x1x64xf32>
    %3022 = stablehlo.sine %3021 : tensor<8x4096x1x64xf32>
    %3023 = stablehlo.convert %3022 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %3024 = stablehlo.cosine %3021 : tensor<8x4096x1x64xf32>
    %3025 = stablehlo.convert %3024 : (tensor<8x4096x1x64xf32>) -> tensor<8x4096x1x64xbf16>
    %3026 = stablehlo.reshape %2961#0 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %3027 = stablehlo.dot_general %2961#1, %3026, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %3028 = stablehlo.transpose %3027, dims = [0, 1, 4, 3, 2] : (tensor<8x9x4096x4096x1xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %3029 = stablehlo.broadcast_in_dim %2961#16, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %3030 = stablehlo.broadcast_in_dim %2961#16, dims = [0, 2] : (tensor<8x4096xi32>) -> tensor<8x1x4096xi32>
    %3031 = stablehlo.broadcast_in_dim %3029, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x4096xi32>
    %3032 = stablehlo.broadcast_in_dim %3030, dims = [0, 1, 2] : (tensor<8x1x4096xi32>) -> tensor<8x4096x4096xi32>
    %3033 = stablehlo.compare  EQ, %3031, %3032,  SIGNED : (tensor<8x4096x4096xi32>, tensor<8x4096x4096xi32>) -> tensor<8x4096x4096xi1>
    %3034 = stablehlo.broadcast_in_dim %3033, dims = [0, 3, 4] : (tensor<8x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %3035 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %3036 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %c_426 = stablehlo.constant dense<0> : tensor<i32>
    %3037 = stablehlo.broadcast_in_dim %c_426, dims = [] : (tensor<i32>) -> tensor<4096x4096xi32>
    %3038 = stablehlo.add %3035, %3037 : tensor<4096x4096xi32>
    %3039 = stablehlo.compare  LE, %3036, %3038,  SIGNED : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %3040 = stablehlo.broadcast_in_dim %3039, dims = [3, 4] : (tensor<4096x4096xi1>) -> tensor<1x1x1x4096x4096xi1>
    %3041 = stablehlo.broadcast_in_dim %3040, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x4096x4096xi1>) -> tensor<8x1x1x4096x4096xi1>
    %3042 = stablehlo.and %3034, %3041 : tensor<8x1x1x4096x4096xi1>
    %cst_427 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_428 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %3043 = call @_where_115(%3042, %cst_427, %cst_428) : (tensor<8x1x1x4096x4096xi1>, tensor<f32>, tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %cst_429 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %3044 = stablehlo.broadcast_in_dim %cst_429, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %3045 = stablehlo.compare  GE, %3043, %3044,  FLOAT : (tensor<8x1x1x4096x4096xf32>, tensor<8x1x1x4096x4096xf32>) -> tensor<8x1x1x4096x4096xi1>
    %cst_430 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %3046:2 = call @_where_116(%3045, %3028, %cst_430) : (tensor<8x1x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>, tensor<f32>) -> (tensor<8x9x1x4096x4096xbf16>, tensor<8x9x1x4096x4096xi1>)
    %3047 = stablehlo.reshape %3046#0 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %cst_431 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3048 = stablehlo.reduce(%3047 init: %cst_431) applies stablehlo.maximum across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %3049 = stablehlo.reshape %3048 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %3050 = stablehlo.broadcast_in_dim %3049, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %3051 = stablehlo.compare  EQ, %3047, %3050,  FLOAT : (tensor<8x9x4096x4096xbf16>, tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xi1>
    %3052 = stablehlo.convert %3051 : (tensor<8x9x4096x4096xi1>) -> tensor<8x9x4096x4096xbf16>
    %cst_432 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3053 = stablehlo.reduce(%3052 init: %cst_432) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %3054 = stablehlo.broadcast_in_dim %3048, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %3055 = stablehlo.broadcast_in_dim %3054, dims = [0, 1, 2, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x4096xbf16>
    %3056 = stablehlo.subtract %3047, %3055 : tensor<8x9x4096x4096xbf16>
    %3057 = stablehlo.exponential %3056 : tensor<8x9x4096x4096xbf16>
    %3058 = stablehlo.convert %3057 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x4096x4096xf32>
    %cst_433 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3059 = stablehlo.reduce(%3058 init: %cst_433) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %3060 = stablehlo.broadcast_in_dim %3059, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x1xf32>
    %3061 = stablehlo.convert %3060 : (tensor<8x9x4096x1xf32>) -> tensor<8x9x4096x1xbf16>
    %3062 = stablehlo.reshape %3057 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %3063 = stablehlo.transpose %3061, dims = [0, 2, 1, 3] : (tensor<8x9x4096x1xbf16>) -> tensor<8x4096x9x1xbf16>
    %3064 = stablehlo.dot_general %2961#2, %3062, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %3065 = stablehlo.transpose %3064, dims = [0, 4, 1, 3, 2] : (tensor<8x9x128x1x4096xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %3066 = stablehlo.reshape %3065 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %3067 = stablehlo.broadcast_in_dim %3063, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %3068 = stablehlo.divide %3066, %3067 : tensor<8x4096x9x128xbf16>
    %3069 = stablehlo.multiply %3063, %3063 : tensor<8x4096x9x1xbf16>
    %cst_434 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3070 = stablehlo.broadcast_in_dim %cst_434, dims = [] : (tensor<bf16>) -> tensor<8x4096x9x1xbf16>
    %3071 = stablehlo.divide %3070, %3069 : tensor<8x4096x9x1xbf16>
    %3072 = stablehlo.custom_call @Sharding(%3068) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %3073 = stablehlo.custom_call @Sharding(%2961#3) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %3074 = stablehlo.add %2962, %3073 : tensor<8x4096x576xbf16>
    %3075 = stablehlo.convert %3074 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %3076 = stablehlo.multiply %3075, %3075 : tensor<8x4096x576xf32>
    %cst_435 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %3077 = stablehlo.broadcast_in_dim %cst_435, dims = [] : (tensor<f32>) -> tensor<8x4096x576xf32>
    %3078 = stablehlo.multiply %3077, %3075 : tensor<8x4096x576xf32>
    %cst_436 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3079 = stablehlo.reduce(%3076 init: %cst_436) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %3080 = stablehlo.broadcast_in_dim %3079, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %cst_437 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %3081 = stablehlo.broadcast_in_dim %cst_437, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %3082 = stablehlo.divide %3080, %3081 : tensor<8x4096x1xf32>
    %cst_438 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3083 = stablehlo.broadcast_in_dim %cst_438, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %3084 = stablehlo.add %3082, %3083 : tensor<8x4096x1xf32>
    %3085 = stablehlo.rsqrt %3084 : tensor<8x4096x1xf32>
    %3086 = stablehlo.divide %3085, %3084 : tensor<8x4096x1xf32>
    %cst_439 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %3087 = stablehlo.broadcast_in_dim %cst_439, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %3088 = stablehlo.multiply %3087, %3086 : tensor<8x4096x1xf32>
    %3089 = stablehlo.broadcast_in_dim %3085, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %3090 = stablehlo.multiply %3075, %3089 : tensor<8x4096x576xf32>
    %3091 = stablehlo.convert %3090 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %3092 = stablehlo.broadcast_in_dim %2961#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %3093 = stablehlo.broadcast_in_dim %3092, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %3094 = stablehlo.multiply %3091, %3093 : tensor<8x4096x576xbf16>
    %3095 = stablehlo.custom_call @Sharding(%3094) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %3096:3 = call @silu_124(%2961#4) : (tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>)
    %3097 = stablehlo.multiply %3096#0, %2961#5 : tensor<8x4096x2016xbf16>
    %3098 = stablehlo.custom_call @Sharding(%3097) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %3099 = stablehlo.custom_call @Sharding(%2961#18) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %3100 = stablehlo.custom_call @Sharding(%3099) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %3101 = stablehlo.dot_general %3100, %3098, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %3102 = stablehlo.transpose %3101, dims = [1, 0] : (tensor<576x2016xbf16>) -> tensor<2016x576xbf16>
    %3103 = stablehlo.dot_general %3100, %2961#8, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<8x4096x2016xbf16>
    %3104 = stablehlo.custom_call @Sharding(%3103) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %3105 = stablehlo.multiply %3096#0, %3104 : tensor<8x4096x2016xbf16>
    %3106 = stablehlo.multiply %3104, %2961#5 : tensor<8x4096x2016xbf16>
    %3107 = stablehlo.dot_general %3105, %3095, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %3108 = stablehlo.transpose %3107, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %3109 = stablehlo.dot_general %3105, %2961#7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %3110 = call @silu_133(%3096#1, %2961#4, %3096#2, %3106) : (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16>
    %3111 = stablehlo.dot_general %3110, %3095, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<8x4096x576xbf16>) -> tensor<2016x576xbf16>
    %3112 = stablehlo.transpose %3111, dims = [1, 0] : (tensor<2016x576xbf16>) -> tensor<576x2016xbf16>
    %3113 = stablehlo.dot_general %3110, %2961#6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<8x4096x576xbf16>
    %3114 = stablehlo.add %3109, %3113 : tensor<8x4096x576xbf16>
    %3115 = stablehlo.custom_call @Sharding(%3114) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %3116 = stablehlo.multiply %3091, %3115 : tensor<8x4096x576xbf16>
    %cst_440 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3117 = stablehlo.reduce(%3116 init: %cst_440) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %3118 = stablehlo.reshape %3117 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %3119 = stablehlo.broadcast_in_dim %3092, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %3120 = stablehlo.multiply %3115, %3119 : tensor<8x4096x576xbf16>
    %cst_441 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3121 = stablehlo.reduce(%3118 init: %cst_441) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %3122 = stablehlo.convert %3120 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %3123 = stablehlo.multiply %3075, %3122 : tensor<8x4096x576xf32>
    %cst_442 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3124 = stablehlo.reduce(%3123 init: %cst_442) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %3125 = stablehlo.reshape %3124 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %3126 = stablehlo.broadcast_in_dim %3085, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %3127 = stablehlo.multiply %3122, %3126 : tensor<8x4096x576xf32>
    %3128 = stablehlo.multiply %3125, %3088 : tensor<8x4096x1xf32>
    %cst_443 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %3129 = stablehlo.broadcast_in_dim %cst_443, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %3130 = stablehlo.divide %3128, %3129 : tensor<8x4096x1xf32>
    %cst_444 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3131 = stablehlo.reduce(%3130 init: %cst_444) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %3132 = stablehlo.broadcast_in_dim %3131, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %3133 = stablehlo.multiply %3132, %3078 : tensor<8x4096x576xf32>
    %3134 = stablehlo.add %3127, %3133 : tensor<8x4096x576xf32>
    %3135 = stablehlo.convert %3134 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %3136 = stablehlo.add %3099, %3135 : tensor<8x4096x576xbf16>
    %3137 = stablehlo.custom_call @Sharding(%3136) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %3138 = stablehlo.dot_general %3137, %3072, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %3139 = stablehlo.transpose %3138, dims = [1, 2, 0] : (tensor<576x9x128xbf16>) -> tensor<9x128x576xbf16>
    %3140 = stablehlo.dot_general %3137, %2961#12, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<8x4096x9x128xbf16>
    %3141 = stablehlo.custom_call @Sharding(%3140) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %3142 = stablehlo.broadcast_in_dim %3071, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %3143 = stablehlo.multiply %3141, %3142 : tensor<8x4096x9x128xbf16>
    %3144 = stablehlo.multiply %3143, %3066 : tensor<8x4096x9x128xbf16>
    %cst_445 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3145 = stablehlo.reduce(%3144 init: %cst_445) applies stablehlo.add across dimensions = [3] : (tensor<8x4096x9x128xbf16>, tensor<bf16>) -> tensor<8x4096x9xbf16>
    %3146 = stablehlo.reshape %3145 : (tensor<8x4096x9xbf16>) -> tensor<8x4096x9x1xbf16>
    %3147 = stablehlo.negate %3146 : tensor<8x4096x9x1xbf16>
    %3148 = stablehlo.broadcast_in_dim %3063, dims = [0, 1, 2, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x4096x9x128xbf16>
    %3149 = stablehlo.divide %3141, %3148 : tensor<8x4096x9x128xbf16>
    %3150 = stablehlo.reshape %3149 : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %3151 = stablehlo.transpose %3150, dims = [0, 2, 4, 3, 1] : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x128x1x4096xbf16>
    %3152 = stablehlo.dot_general %3151, %2961#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %3153 = stablehlo.dot_general %3151, %3062, batching_dims = [0, 1] x [0, 1], contracting_dims = [3, 4] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x128x1x4096xbf16>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x128x4096xbf16>
    %3154 = stablehlo.transpose %3153, dims = [0, 3, 1, 2] : (tensor<8x9x128x4096xbf16>) -> tensor<8x4096x9x128xbf16>
    %3155 = stablehlo.transpose %3147, dims = [0, 2, 1, 3] : (tensor<8x4096x9x1xbf16>) -> tensor<8x9x4096x1xbf16>
    %3156 = stablehlo.reshape %3152 : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %3157 = stablehlo.convert %3155 : (tensor<8x9x4096x1xbf16>) -> tensor<8x9x4096x1xf32>
    %cst_446 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3158 = stablehlo.reduce(%3157 init: %cst_446) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xf32>, tensor<f32>) -> tensor<8x9x4096xf32>
    %3159 = stablehlo.broadcast_in_dim %3158, dims = [0, 1, 2] : (tensor<8x9x4096xf32>) -> tensor<8x9x4096x4096xf32>
    %3160 = stablehlo.convert %3159 : (tensor<8x9x4096x4096xf32>) -> tensor<8x9x4096x4096xbf16>
    %3161 = stablehlo.add %3156, %3160 : tensor<8x9x4096x4096xbf16>
    %3162 = stablehlo.multiply %3161, %3057 : tensor<8x9x4096x4096xbf16>
    %3163 = stablehlo.negate %3162 : tensor<8x9x4096x4096xbf16>
    %cst_447 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3164 = stablehlo.reduce(%3163 init: %cst_447) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x4096xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %3165 = stablehlo.reshape %3164 : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x1xbf16>
    %cst_448 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3166 = stablehlo.reduce(%3165 init: %cst_448) applies stablehlo.add across dimensions = [3] : (tensor<8x9x4096x1xbf16>, tensor<bf16>) -> tensor<8x9x4096xbf16>
    %3167 = stablehlo.divide %3166, %3053 : tensor<8x9x4096xbf16>
    %3168 = stablehlo.broadcast_in_dim %3167, dims = [0, 1, 2] : (tensor<8x9x4096xbf16>) -> tensor<8x9x4096x4096xbf16>
    %3169 = stablehlo.multiply %3168, %3052 : tensor<8x9x4096x4096xbf16>
    %3170 = stablehlo.add %3162, %3169 : tensor<8x9x4096x4096xbf16>
    %3171 = stablehlo.reshape %3170 : (tensor<8x9x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %3172 = call @_where_163(%3046#1, %3171) : (tensor<8x9x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16>
    %3173 = stablehlo.transpose %3172, dims = [0, 1, 4, 3, 2] : (tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x4096x4096x1xbf16>
    %3174 = stablehlo.dot_general %3173, %2961#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x128xbf16>) -> tensor<8x9x4096x1x128xbf16>
    %3175 = stablehlo.transpose %3174, dims = [0, 2, 1, 3, 4] : (tensor<8x9x4096x1x128xbf16>) -> tensor<8x4096x9x1x128xbf16>
    %3176 = stablehlo.dot_general %3173, %3026, batching_dims = [0, 1] x [0, 2], contracting_dims = [3, 4] x [1, 3], precision = [DEFAULT, DEFAULT] : (tensor<8x9x4096x4096x1xbf16>, tensor<8x4096x9x1x128xbf16>) -> tensor<8x9x4096x128xbf16>
    %3177 = stablehlo.transpose %3176, dims = [0, 2, 1, 3] : (tensor<8x9x4096x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %3178 = stablehlo.reshape %3175 : (tensor<8x4096x9x1x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %3179 = stablehlo.custom_call @Sharding(%3154) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %3180 = stablehlo.custom_call @Sharding(%3177) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %3181 = stablehlo.custom_call @Sharding(%3178) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x128xbf16>
    %3182 = stablehlo.slice %3180 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %3183 = stablehlo.slice %3180 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %3184 = stablehlo.broadcast_in_dim %3023, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %3185 = stablehlo.multiply %3183, %3184 : tensor<8x4096x9x64xbf16>
    %3186 = stablehlo.broadcast_in_dim %3025, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %3187 = stablehlo.multiply %3183, %3186 : tensor<8x4096x9x64xbf16>
    %3188 = stablehlo.negate %3182 : tensor<8x4096x9x64xbf16>
    %3189 = stablehlo.broadcast_in_dim %3023, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %3190 = stablehlo.multiply %3188, %3189 : tensor<8x4096x9x64xbf16>
    %3191 = stablehlo.add %3187, %3190 : tensor<8x4096x9x64xbf16>
    %3192 = stablehlo.broadcast_in_dim %3025, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %3193 = stablehlo.multiply %3182, %3192 : tensor<8x4096x9x64xbf16>
    %3194 = stablehlo.add %3185, %3193 : tensor<8x4096x9x64xbf16>
    %3195 = stablehlo.concatenate %3194, %3191, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %3196 = stablehlo.slice %3181 [0:8, 0:4096, 0:9, 0:64] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %3197 = stablehlo.slice %3181 [0:8, 0:4096, 0:9, 64:128] : (tensor<8x4096x9x128xbf16>) -> tensor<8x4096x9x64xbf16>
    %3198 = stablehlo.broadcast_in_dim %3003, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %3199 = stablehlo.multiply %3197, %3198 : tensor<8x4096x9x64xbf16>
    %3200 = stablehlo.broadcast_in_dim %3005, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %3201 = stablehlo.multiply %3197, %3200 : tensor<8x4096x9x64xbf16>
    %3202 = stablehlo.negate %3196 : tensor<8x4096x9x64xbf16>
    %3203 = stablehlo.broadcast_in_dim %3003, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %3204 = stablehlo.multiply %3202, %3203 : tensor<8x4096x9x64xbf16>
    %3205 = stablehlo.add %3201, %3204 : tensor<8x4096x9x64xbf16>
    %3206 = stablehlo.broadcast_in_dim %3005, dims = [0, 1, 2, 3] : (tensor<8x4096x1x64xbf16>) -> tensor<8x4096x9x64xbf16>
    %3207 = stablehlo.multiply %3196, %3206 : tensor<8x4096x9x64xbf16>
    %3208 = stablehlo.add %3199, %3207 : tensor<8x4096x9x64xbf16>
    %3209 = stablehlo.concatenate %3208, %3205, dim = 3 : (tensor<8x4096x9x64xbf16>, tensor<8x4096x9x64xbf16>) -> tensor<8x4096x9x128xbf16>
    %3210 = stablehlo.dot_general %3179, %2985, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %3211 = stablehlo.transpose %3210, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %3212 = stablehlo.dot_general %3179, %2961#14, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %3213 = stablehlo.dot_general %3195, %2985, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %3214 = stablehlo.transpose %3213, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %3215 = stablehlo.dot_general %3195, %2961#11, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %3216 = stablehlo.add %3212, %3215 : tensor<8x4096x576xbf16>
    %3217 = stablehlo.dot_general %3209, %2984, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<8x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %3218 = stablehlo.transpose %3217, dims = [2, 0, 1] : (tensor<9x128x576xbf16>) -> tensor<576x9x128xbf16>
    %3219 = stablehlo.dot_general %3209, %2961#13, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<8x4096x576xbf16>
    %3220 = stablehlo.custom_call @Sharding(%3216) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %3221 = stablehlo.custom_call @Sharding(%3219) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %3222 = stablehlo.add %3220, %3221 : tensor<8x4096x576xbf16>
    %3223 = stablehlo.custom_call @Sharding(%3222) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %3224 = stablehlo.multiply %2979, %3223 : tensor<8x4096x576xbf16>
    %cst_449 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3225 = stablehlo.reduce(%3224 init: %cst_449) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %3226 = stablehlo.reshape %3225 : (tensor<576xbf16>) -> tensor<1x1x576xbf16>
    %3227 = stablehlo.broadcast_in_dim %2980, dims = [0, 1, 2] : (tensor<1x1x576xbf16>) -> tensor<8x4096x576xbf16>
    %3228 = stablehlo.multiply %3223, %3227 : tensor<8x4096x576xbf16>
    %cst_450 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3229 = stablehlo.reduce(%3226 init: %cst_450) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %3230 = stablehlo.convert %3228 : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xf32>
    %3231 = stablehlo.multiply %2963, %3230 : tensor<8x4096x576xf32>
    %cst_451 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3232 = stablehlo.reduce(%3231 init: %cst_451) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x576xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %3233 = stablehlo.reshape %3232 : (tensor<8x4096xf32>) -> tensor<8x4096x1xf32>
    %3234 = stablehlo.broadcast_in_dim %2973, dims = [0, 1, 2] : (tensor<8x4096x1xf32>) -> tensor<8x4096x576xf32>
    %3235 = stablehlo.multiply %3230, %3234 : tensor<8x4096x576xf32>
    %3236 = stablehlo.multiply %3233, %2976 : tensor<8x4096x1xf32>
    %cst_452 = stablehlo.constant dense<5.760000e+02> : tensor<f32>
    %3237 = stablehlo.broadcast_in_dim %cst_452, dims = [] : (tensor<f32>) -> tensor<8x4096x1xf32>
    %3238 = stablehlo.divide %3236, %3237 : tensor<8x4096x1xf32>
    %cst_453 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3239 = stablehlo.reduce(%3238 init: %cst_453) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x1xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %3240 = stablehlo.broadcast_in_dim %3239, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x576xf32>
    %3241 = stablehlo.multiply %3240, %2966 : tensor<8x4096x576xf32>
    %3242 = stablehlo.add %3235, %3241 : tensor<8x4096x576xf32>
    %3243 = stablehlo.convert %3242 : (tensor<8x4096x576xf32>) -> tensor<8x4096x576xbf16>
    %3244 = stablehlo.add %3136, %3243 : tensor<8x4096x576xbf16>
    %3245 = stablehlo.custom_call @Sharding(%3244) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %3246 = stablehlo.custom_call @Sharding(%3245) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<8x4096x576xbf16>) -> tensor<8x4096x576xbf16>
    %3247 = stablehlo.dot_general %3246, %6, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<8x4096x576xbf16>, tensor<8x4096x128256xbf16>) -> tensor<576x128256xbf16>
    %3248 = stablehlo.transpose %3247, dims = [1, 0] : (tensor<576x128256xbf16>) -> tensor<128256x576xbf16>
    %3249 = stablehlo.multiply %1236, %1236 : tensor<576xbf16>
    %3250 = stablehlo.convert %3249 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_454 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3251 = stablehlo.reduce(%3250 init: %cst_454) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3252 = stablehlo.convert %3251 : (tensor<f32>) -> tensor<bf16>
    %cst_455 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3253 = stablehlo.add %cst_455, %3252 : tensor<bf16>
    %3254 = stablehlo.multiply %3112, %3112 : tensor<576x2016xbf16>
    %3255 = stablehlo.convert %3254 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_456 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3256 = stablehlo.reduce(%3255 init: %cst_456) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3257 = stablehlo.convert %3256 : (tensor<f32>) -> tensor<bf16>
    %3258 = stablehlo.add %3253, %3257 : tensor<bf16>
    %3259 = stablehlo.multiply %3108, %3108 : tensor<576x2016xbf16>
    %3260 = stablehlo.convert %3259 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_457 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3261 = stablehlo.reduce(%3260 init: %cst_457) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3262 = stablehlo.convert %3261 : (tensor<f32>) -> tensor<bf16>
    %3263 = stablehlo.add %3258, %3262 : tensor<bf16>
    %3264 = stablehlo.multiply %3102, %3102 : tensor<2016x576xbf16>
    %3265 = stablehlo.convert %3264 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_458 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3266 = stablehlo.reduce(%3265 init: %cst_458) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %3267 = stablehlo.convert %3266 : (tensor<f32>) -> tensor<bf16>
    %3268 = stablehlo.add %3263, %3267 : tensor<bf16>
    %3269 = stablehlo.multiply %3121, %3121 : tensor<576xbf16>
    %3270 = stablehlo.convert %3269 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_459 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3271 = stablehlo.reduce(%3270 init: %cst_459) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3272 = stablehlo.convert %3271 : (tensor<f32>) -> tensor<bf16>
    %3273 = stablehlo.add %3268, %3272 : tensor<bf16>
    %3274 = stablehlo.multiply %3229, %3229 : tensor<576xbf16>
    %3275 = stablehlo.convert %3274 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_460 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3276 = stablehlo.reduce(%3275 init: %cst_460) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3277 = stablehlo.convert %3276 : (tensor<f32>) -> tensor<bf16>
    %3278 = stablehlo.add %3273, %3277 : tensor<bf16>
    %3279 = stablehlo.multiply %3214, %3214 : tensor<576x9x128xbf16>
    %3280 = stablehlo.convert %3279 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_461 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3281 = stablehlo.reduce(%3280 init: %cst_461) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3282 = stablehlo.convert %3281 : (tensor<f32>) -> tensor<bf16>
    %3283 = stablehlo.add %3278, %3282 : tensor<bf16>
    %3284 = stablehlo.multiply %3139, %3139 : tensor<9x128x576xbf16>
    %3285 = stablehlo.convert %3284 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_462 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3286 = stablehlo.reduce(%3285 init: %cst_462) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %3287 = stablehlo.convert %3286 : (tensor<f32>) -> tensor<bf16>
    %3288 = stablehlo.add %3283, %3287 : tensor<bf16>
    %3289 = stablehlo.multiply %3218, %3218 : tensor<576x9x128xbf16>
    %3290 = stablehlo.convert %3289 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_463 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3291 = stablehlo.reduce(%3290 init: %cst_463) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3292 = stablehlo.convert %3291 : (tensor<f32>) -> tensor<bf16>
    %3293 = stablehlo.add %3288, %3292 : tensor<bf16>
    %3294 = stablehlo.multiply %3211, %3211 : tensor<576x9x128xbf16>
    %3295 = stablehlo.convert %3294 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_464 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3296 = stablehlo.reduce(%3295 init: %cst_464) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3297 = stablehlo.convert %3296 : (tensor<f32>) -> tensor<bf16>
    %3298 = stablehlo.add %3293, %3297 : tensor<bf16>
    %3299 = stablehlo.multiply %2827, %2827 : tensor<576x2016xbf16>
    %3300 = stablehlo.convert %3299 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_465 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3301 = stablehlo.reduce(%3300 init: %cst_465) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3302 = stablehlo.convert %3301 : (tensor<f32>) -> tensor<bf16>
    %3303 = stablehlo.add %3298, %3302 : tensor<bf16>
    %3304 = stablehlo.multiply %2823, %2823 : tensor<576x2016xbf16>
    %3305 = stablehlo.convert %3304 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_466 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3306 = stablehlo.reduce(%3305 init: %cst_466) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3307 = stablehlo.convert %3306 : (tensor<f32>) -> tensor<bf16>
    %3308 = stablehlo.add %3303, %3307 : tensor<bf16>
    %3309 = stablehlo.multiply %2817, %2817 : tensor<2016x576xbf16>
    %3310 = stablehlo.convert %3309 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_467 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3311 = stablehlo.reduce(%3310 init: %cst_467) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %3312 = stablehlo.convert %3311 : (tensor<f32>) -> tensor<bf16>
    %3313 = stablehlo.add %3308, %3312 : tensor<bf16>
    %3314 = stablehlo.multiply %2836, %2836 : tensor<576xbf16>
    %3315 = stablehlo.convert %3314 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_468 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3316 = stablehlo.reduce(%3315 init: %cst_468) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3317 = stablehlo.convert %3316 : (tensor<f32>) -> tensor<bf16>
    %3318 = stablehlo.add %3313, %3317 : tensor<bf16>
    %3319 = stablehlo.multiply %2944, %2944 : tensor<576xbf16>
    %3320 = stablehlo.convert %3319 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_469 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3321 = stablehlo.reduce(%3320 init: %cst_469) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3322 = stablehlo.convert %3321 : (tensor<f32>) -> tensor<bf16>
    %3323 = stablehlo.add %3318, %3322 : tensor<bf16>
    %3324 = stablehlo.multiply %2929, %2929 : tensor<576x9x128xbf16>
    %3325 = stablehlo.convert %3324 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_470 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3326 = stablehlo.reduce(%3325 init: %cst_470) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3327 = stablehlo.convert %3326 : (tensor<f32>) -> tensor<bf16>
    %3328 = stablehlo.add %3323, %3327 : tensor<bf16>
    %3329 = stablehlo.multiply %2854, %2854 : tensor<9x128x576xbf16>
    %3330 = stablehlo.convert %3329 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_471 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3331 = stablehlo.reduce(%3330 init: %cst_471) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %3332 = stablehlo.convert %3331 : (tensor<f32>) -> tensor<bf16>
    %3333 = stablehlo.add %3328, %3332 : tensor<bf16>
    %3334 = stablehlo.multiply %2933, %2933 : tensor<576x9x128xbf16>
    %3335 = stablehlo.convert %3334 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_472 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3336 = stablehlo.reduce(%3335 init: %cst_472) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3337 = stablehlo.convert %3336 : (tensor<f32>) -> tensor<bf16>
    %3338 = stablehlo.add %3333, %3337 : tensor<bf16>
    %3339 = stablehlo.multiply %2926, %2926 : tensor<576x9x128xbf16>
    %3340 = stablehlo.convert %3339 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_473 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3341 = stablehlo.reduce(%3340 init: %cst_473) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3342 = stablehlo.convert %3341 : (tensor<f32>) -> tensor<bf16>
    %3343 = stablehlo.add %3338, %3342 : tensor<bf16>
    %3344 = stablehlo.multiply %2542, %2542 : tensor<576x2016xbf16>
    %3345 = stablehlo.convert %3344 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_474 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3346 = stablehlo.reduce(%3345 init: %cst_474) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3347 = stablehlo.convert %3346 : (tensor<f32>) -> tensor<bf16>
    %3348 = stablehlo.add %3343, %3347 : tensor<bf16>
    %3349 = stablehlo.multiply %2538, %2538 : tensor<576x2016xbf16>
    %3350 = stablehlo.convert %3349 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_475 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3351 = stablehlo.reduce(%3350 init: %cst_475) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3352 = stablehlo.convert %3351 : (tensor<f32>) -> tensor<bf16>
    %3353 = stablehlo.add %3348, %3352 : tensor<bf16>
    %3354 = stablehlo.multiply %2532, %2532 : tensor<2016x576xbf16>
    %3355 = stablehlo.convert %3354 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_476 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3356 = stablehlo.reduce(%3355 init: %cst_476) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %3357 = stablehlo.convert %3356 : (tensor<f32>) -> tensor<bf16>
    %3358 = stablehlo.add %3353, %3357 : tensor<bf16>
    %3359 = stablehlo.multiply %2551, %2551 : tensor<576xbf16>
    %3360 = stablehlo.convert %3359 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_477 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3361 = stablehlo.reduce(%3360 init: %cst_477) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3362 = stablehlo.convert %3361 : (tensor<f32>) -> tensor<bf16>
    %3363 = stablehlo.add %3358, %3362 : tensor<bf16>
    %3364 = stablehlo.multiply %2659, %2659 : tensor<576xbf16>
    %3365 = stablehlo.convert %3364 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_478 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3366 = stablehlo.reduce(%3365 init: %cst_478) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3367 = stablehlo.convert %3366 : (tensor<f32>) -> tensor<bf16>
    %3368 = stablehlo.add %3363, %3367 : tensor<bf16>
    %3369 = stablehlo.multiply %2644, %2644 : tensor<576x9x128xbf16>
    %3370 = stablehlo.convert %3369 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_479 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3371 = stablehlo.reduce(%3370 init: %cst_479) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3372 = stablehlo.convert %3371 : (tensor<f32>) -> tensor<bf16>
    %3373 = stablehlo.add %3368, %3372 : tensor<bf16>
    %3374 = stablehlo.multiply %2569, %2569 : tensor<9x128x576xbf16>
    %3375 = stablehlo.convert %3374 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_480 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3376 = stablehlo.reduce(%3375 init: %cst_480) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %3377 = stablehlo.convert %3376 : (tensor<f32>) -> tensor<bf16>
    %3378 = stablehlo.add %3373, %3377 : tensor<bf16>
    %3379 = stablehlo.multiply %2648, %2648 : tensor<576x9x128xbf16>
    %3380 = stablehlo.convert %3379 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_481 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3381 = stablehlo.reduce(%3380 init: %cst_481) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3382 = stablehlo.convert %3381 : (tensor<f32>) -> tensor<bf16>
    %3383 = stablehlo.add %3378, %3382 : tensor<bf16>
    %3384 = stablehlo.multiply %2641, %2641 : tensor<576x9x128xbf16>
    %3385 = stablehlo.convert %3384 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_482 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3386 = stablehlo.reduce(%3385 init: %cst_482) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3387 = stablehlo.convert %3386 : (tensor<f32>) -> tensor<bf16>
    %3388 = stablehlo.add %3383, %3387 : tensor<bf16>
    %3389 = stablehlo.multiply %2257, %2257 : tensor<576x2016xbf16>
    %3390 = stablehlo.convert %3389 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_483 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3391 = stablehlo.reduce(%3390 init: %cst_483) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3392 = stablehlo.convert %3391 : (tensor<f32>) -> tensor<bf16>
    %3393 = stablehlo.add %3388, %3392 : tensor<bf16>
    %3394 = stablehlo.multiply %2253, %2253 : tensor<576x2016xbf16>
    %3395 = stablehlo.convert %3394 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_484 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3396 = stablehlo.reduce(%3395 init: %cst_484) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3397 = stablehlo.convert %3396 : (tensor<f32>) -> tensor<bf16>
    %3398 = stablehlo.add %3393, %3397 : tensor<bf16>
    %3399 = stablehlo.multiply %2247, %2247 : tensor<2016x576xbf16>
    %3400 = stablehlo.convert %3399 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_485 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3401 = stablehlo.reduce(%3400 init: %cst_485) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %3402 = stablehlo.convert %3401 : (tensor<f32>) -> tensor<bf16>
    %3403 = stablehlo.add %3398, %3402 : tensor<bf16>
    %3404 = stablehlo.multiply %2266, %2266 : tensor<576xbf16>
    %3405 = stablehlo.convert %3404 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_486 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3406 = stablehlo.reduce(%3405 init: %cst_486) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3407 = stablehlo.convert %3406 : (tensor<f32>) -> tensor<bf16>
    %3408 = stablehlo.add %3403, %3407 : tensor<bf16>
    %3409 = stablehlo.multiply %2374, %2374 : tensor<576xbf16>
    %3410 = stablehlo.convert %3409 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_487 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3411 = stablehlo.reduce(%3410 init: %cst_487) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3412 = stablehlo.convert %3411 : (tensor<f32>) -> tensor<bf16>
    %3413 = stablehlo.add %3408, %3412 : tensor<bf16>
    %3414 = stablehlo.multiply %2359, %2359 : tensor<576x9x128xbf16>
    %3415 = stablehlo.convert %3414 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_488 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3416 = stablehlo.reduce(%3415 init: %cst_488) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3417 = stablehlo.convert %3416 : (tensor<f32>) -> tensor<bf16>
    %3418 = stablehlo.add %3413, %3417 : tensor<bf16>
    %3419 = stablehlo.multiply %2284, %2284 : tensor<9x128x576xbf16>
    %3420 = stablehlo.convert %3419 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_489 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3421 = stablehlo.reduce(%3420 init: %cst_489) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %3422 = stablehlo.convert %3421 : (tensor<f32>) -> tensor<bf16>
    %3423 = stablehlo.add %3418, %3422 : tensor<bf16>
    %3424 = stablehlo.multiply %2363, %2363 : tensor<576x9x128xbf16>
    %3425 = stablehlo.convert %3424 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_490 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3426 = stablehlo.reduce(%3425 init: %cst_490) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3427 = stablehlo.convert %3426 : (tensor<f32>) -> tensor<bf16>
    %3428 = stablehlo.add %3423, %3427 : tensor<bf16>
    %3429 = stablehlo.multiply %2356, %2356 : tensor<576x9x128xbf16>
    %3430 = stablehlo.convert %3429 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_491 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3431 = stablehlo.reduce(%3430 init: %cst_491) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3432 = stablehlo.convert %3431 : (tensor<f32>) -> tensor<bf16>
    %3433 = stablehlo.add %3428, %3432 : tensor<bf16>
    %3434 = stablehlo.multiply %1972, %1972 : tensor<576x2016xbf16>
    %3435 = stablehlo.convert %3434 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_492 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3436 = stablehlo.reduce(%3435 init: %cst_492) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3437 = stablehlo.convert %3436 : (tensor<f32>) -> tensor<bf16>
    %3438 = stablehlo.add %3433, %3437 : tensor<bf16>
    %3439 = stablehlo.multiply %1968, %1968 : tensor<576x2016xbf16>
    %3440 = stablehlo.convert %3439 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_493 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3441 = stablehlo.reduce(%3440 init: %cst_493) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3442 = stablehlo.convert %3441 : (tensor<f32>) -> tensor<bf16>
    %3443 = stablehlo.add %3438, %3442 : tensor<bf16>
    %3444 = stablehlo.multiply %1962, %1962 : tensor<2016x576xbf16>
    %3445 = stablehlo.convert %3444 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_494 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3446 = stablehlo.reduce(%3445 init: %cst_494) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %3447 = stablehlo.convert %3446 : (tensor<f32>) -> tensor<bf16>
    %3448 = stablehlo.add %3443, %3447 : tensor<bf16>
    %3449 = stablehlo.multiply %1981, %1981 : tensor<576xbf16>
    %3450 = stablehlo.convert %3449 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_495 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3451 = stablehlo.reduce(%3450 init: %cst_495) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3452 = stablehlo.convert %3451 : (tensor<f32>) -> tensor<bf16>
    %3453 = stablehlo.add %3448, %3452 : tensor<bf16>
    %3454 = stablehlo.multiply %2089, %2089 : tensor<576xbf16>
    %3455 = stablehlo.convert %3454 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_496 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3456 = stablehlo.reduce(%3455 init: %cst_496) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3457 = stablehlo.convert %3456 : (tensor<f32>) -> tensor<bf16>
    %3458 = stablehlo.add %3453, %3457 : tensor<bf16>
    %3459 = stablehlo.multiply %2074, %2074 : tensor<576x9x128xbf16>
    %3460 = stablehlo.convert %3459 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_497 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3461 = stablehlo.reduce(%3460 init: %cst_497) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3462 = stablehlo.convert %3461 : (tensor<f32>) -> tensor<bf16>
    %3463 = stablehlo.add %3458, %3462 : tensor<bf16>
    %3464 = stablehlo.multiply %1999, %1999 : tensor<9x128x576xbf16>
    %3465 = stablehlo.convert %3464 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_498 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3466 = stablehlo.reduce(%3465 init: %cst_498) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %3467 = stablehlo.convert %3466 : (tensor<f32>) -> tensor<bf16>
    %3468 = stablehlo.add %3463, %3467 : tensor<bf16>
    %3469 = stablehlo.multiply %2078, %2078 : tensor<576x9x128xbf16>
    %3470 = stablehlo.convert %3469 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_499 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3471 = stablehlo.reduce(%3470 init: %cst_499) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3472 = stablehlo.convert %3471 : (tensor<f32>) -> tensor<bf16>
    %3473 = stablehlo.add %3468, %3472 : tensor<bf16>
    %3474 = stablehlo.multiply %2071, %2071 : tensor<576x9x128xbf16>
    %3475 = stablehlo.convert %3474 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_500 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3476 = stablehlo.reduce(%3475 init: %cst_500) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3477 = stablehlo.convert %3476 : (tensor<f32>) -> tensor<bf16>
    %3478 = stablehlo.add %3473, %3477 : tensor<bf16>
    %3479 = stablehlo.multiply %1687, %1687 : tensor<576x2016xbf16>
    %3480 = stablehlo.convert %3479 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_501 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3481 = stablehlo.reduce(%3480 init: %cst_501) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3482 = stablehlo.convert %3481 : (tensor<f32>) -> tensor<bf16>
    %3483 = stablehlo.add %3478, %3482 : tensor<bf16>
    %3484 = stablehlo.multiply %1683, %1683 : tensor<576x2016xbf16>
    %3485 = stablehlo.convert %3484 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_502 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3486 = stablehlo.reduce(%3485 init: %cst_502) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3487 = stablehlo.convert %3486 : (tensor<f32>) -> tensor<bf16>
    %3488 = stablehlo.add %3483, %3487 : tensor<bf16>
    %3489 = stablehlo.multiply %1677, %1677 : tensor<2016x576xbf16>
    %3490 = stablehlo.convert %3489 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_503 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3491 = stablehlo.reduce(%3490 init: %cst_503) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %3492 = stablehlo.convert %3491 : (tensor<f32>) -> tensor<bf16>
    %3493 = stablehlo.add %3488, %3492 : tensor<bf16>
    %3494 = stablehlo.multiply %1696, %1696 : tensor<576xbf16>
    %3495 = stablehlo.convert %3494 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_504 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3496 = stablehlo.reduce(%3495 init: %cst_504) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3497 = stablehlo.convert %3496 : (tensor<f32>) -> tensor<bf16>
    %3498 = stablehlo.add %3493, %3497 : tensor<bf16>
    %3499 = stablehlo.multiply %1804, %1804 : tensor<576xbf16>
    %3500 = stablehlo.convert %3499 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_505 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3501 = stablehlo.reduce(%3500 init: %cst_505) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3502 = stablehlo.convert %3501 : (tensor<f32>) -> tensor<bf16>
    %3503 = stablehlo.add %3498, %3502 : tensor<bf16>
    %3504 = stablehlo.multiply %1789, %1789 : tensor<576x9x128xbf16>
    %3505 = stablehlo.convert %3504 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_506 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3506 = stablehlo.reduce(%3505 init: %cst_506) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3507 = stablehlo.convert %3506 : (tensor<f32>) -> tensor<bf16>
    %3508 = stablehlo.add %3503, %3507 : tensor<bf16>
    %3509 = stablehlo.multiply %1714, %1714 : tensor<9x128x576xbf16>
    %3510 = stablehlo.convert %3509 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_507 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3511 = stablehlo.reduce(%3510 init: %cst_507) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %3512 = stablehlo.convert %3511 : (tensor<f32>) -> tensor<bf16>
    %3513 = stablehlo.add %3508, %3512 : tensor<bf16>
    %3514 = stablehlo.multiply %1793, %1793 : tensor<576x9x128xbf16>
    %3515 = stablehlo.convert %3514 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_508 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3516 = stablehlo.reduce(%3515 init: %cst_508) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3517 = stablehlo.convert %3516 : (tensor<f32>) -> tensor<bf16>
    %3518 = stablehlo.add %3513, %3517 : tensor<bf16>
    %3519 = stablehlo.multiply %1786, %1786 : tensor<576x9x128xbf16>
    %3520 = stablehlo.convert %3519 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_509 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3521 = stablehlo.reduce(%3520 init: %cst_509) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3522 = stablehlo.convert %3521 : (tensor<f32>) -> tensor<bf16>
    %3523 = stablehlo.add %3518, %3522 : tensor<bf16>
    %3524 = stablehlo.multiply %1402, %1402 : tensor<576x2016xbf16>
    %3525 = stablehlo.convert %3524 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_510 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3526 = stablehlo.reduce(%3525 init: %cst_510) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3527 = stablehlo.convert %3526 : (tensor<f32>) -> tensor<bf16>
    %3528 = stablehlo.add %3523, %3527 : tensor<bf16>
    %3529 = stablehlo.multiply %1398, %1398 : tensor<576x2016xbf16>
    %3530 = stablehlo.convert %3529 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_511 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3531 = stablehlo.reduce(%3530 init: %cst_511) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %3532 = stablehlo.convert %3531 : (tensor<f32>) -> tensor<bf16>
    %3533 = stablehlo.add %3528, %3532 : tensor<bf16>
    %3534 = stablehlo.multiply %1392, %1392 : tensor<2016x576xbf16>
    %3535 = stablehlo.convert %3534 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_512 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3536 = stablehlo.reduce(%3535 init: %cst_512) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %3537 = stablehlo.convert %3536 : (tensor<f32>) -> tensor<bf16>
    %3538 = stablehlo.add %3533, %3537 : tensor<bf16>
    %3539 = stablehlo.multiply %1411, %1411 : tensor<576xbf16>
    %3540 = stablehlo.convert %3539 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_513 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3541 = stablehlo.reduce(%3540 init: %cst_513) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3542 = stablehlo.convert %3541 : (tensor<f32>) -> tensor<bf16>
    %3543 = stablehlo.add %3538, %3542 : tensor<bf16>
    %3544 = stablehlo.multiply %1519, %1519 : tensor<576xbf16>
    %3545 = stablehlo.convert %3544 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_514 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3546 = stablehlo.reduce(%3545 init: %cst_514) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %3547 = stablehlo.convert %3546 : (tensor<f32>) -> tensor<bf16>
    %3548 = stablehlo.add %3543, %3547 : tensor<bf16>
    %3549 = stablehlo.multiply %1504, %1504 : tensor<576x9x128xbf16>
    %3550 = stablehlo.convert %3549 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_515 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3551 = stablehlo.reduce(%3550 init: %cst_515) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3552 = stablehlo.convert %3551 : (tensor<f32>) -> tensor<bf16>
    %3553 = stablehlo.add %3548, %3552 : tensor<bf16>
    %3554 = stablehlo.multiply %1429, %1429 : tensor<9x128x576xbf16>
    %3555 = stablehlo.convert %3554 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_516 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3556 = stablehlo.reduce(%3555 init: %cst_516) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %3557 = stablehlo.convert %3556 : (tensor<f32>) -> tensor<bf16>
    %3558 = stablehlo.add %3553, %3557 : tensor<bf16>
    %3559 = stablehlo.multiply %1508, %1508 : tensor<576x9x128xbf16>
    %3560 = stablehlo.convert %3559 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_517 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3561 = stablehlo.reduce(%3560 init: %cst_517) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3562 = stablehlo.convert %3561 : (tensor<f32>) -> tensor<bf16>
    %3563 = stablehlo.add %3558, %3562 : tensor<bf16>
    %3564 = stablehlo.multiply %1501, %1501 : tensor<576x9x128xbf16>
    %3565 = stablehlo.convert %3564 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_518 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3566 = stablehlo.reduce(%3565 init: %cst_518) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %3567 = stablehlo.convert %3566 : (tensor<f32>) -> tensor<bf16>
    %3568 = stablehlo.add %3563, %3567 : tensor<bf16>
    %3569 = stablehlo.multiply %1229, %1229 : tensor<576x128256xbf16>
    %3570 = stablehlo.convert %3569 : (tensor<576x128256xbf16>) -> tensor<576x128256xf32>
    %cst_519 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3571 = stablehlo.reduce(%3570 init: %cst_519) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x128256xf32>, tensor<f32>) -> tensor<f32>
    %3572 = stablehlo.convert %3571 : (tensor<f32>) -> tensor<bf16>
    %3573 = stablehlo.add %3568, %3572 : tensor<bf16>
    %3574 = stablehlo.multiply %3248, %3248 : tensor<128256x576xbf16>
    %3575 = stablehlo.convert %3574 : (tensor<128256x576xbf16>) -> tensor<128256x576xf32>
    %cst_520 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3576 = stablehlo.reduce(%3575 init: %cst_520) applies stablehlo.add across dimensions = [0, 1] : (tensor<128256x576xf32>, tensor<f32>) -> tensor<f32>
    %3577 = stablehlo.convert %3576 : (tensor<f32>) -> tensor<bf16>
    %3578 = stablehlo.add %3573, %3577 : tensor<bf16>
    %3579 = stablehlo.sqrt %3578 : tensor<bf16>
    %cst_521 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3580 = stablehlo.compare  LT, %3579, %cst_521,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
    %3581 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3582 = stablehlo.divide %1236, %3581 : tensor<576xbf16>
    %cst_522 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3583 = stablehlo.broadcast_in_dim %cst_522, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3584 = stablehlo.multiply %3582, %3583 : tensor<576xbf16>
    %3585 = stablehlo.select %3580, %1236, %3584 : tensor<i1>, tensor<576xbf16>
    %3586 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3587 = stablehlo.divide %3112, %3586 : tensor<576x2016xbf16>
    %cst_523 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3588 = stablehlo.broadcast_in_dim %cst_523, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3589 = stablehlo.multiply %3587, %3588 : tensor<576x2016xbf16>
    %3590 = stablehlo.select %3580, %3112, %3589 : tensor<i1>, tensor<576x2016xbf16>
    %3591 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3592 = stablehlo.divide %3108, %3591 : tensor<576x2016xbf16>
    %cst_524 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3593 = stablehlo.broadcast_in_dim %cst_524, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3594 = stablehlo.multiply %3592, %3593 : tensor<576x2016xbf16>
    %3595 = stablehlo.select %3580, %3108, %3594 : tensor<i1>, tensor<576x2016xbf16>
    %3596 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3597 = stablehlo.divide %3102, %3596 : tensor<2016x576xbf16>
    %cst_525 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3598 = stablehlo.broadcast_in_dim %cst_525, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3599 = stablehlo.multiply %3597, %3598 : tensor<2016x576xbf16>
    %3600 = stablehlo.select %3580, %3102, %3599 : tensor<i1>, tensor<2016x576xbf16>
    %3601 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3602 = stablehlo.divide %3121, %3601 : tensor<576xbf16>
    %cst_526 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3603 = stablehlo.broadcast_in_dim %cst_526, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3604 = stablehlo.multiply %3602, %3603 : tensor<576xbf16>
    %3605 = stablehlo.select %3580, %3121, %3604 : tensor<i1>, tensor<576xbf16>
    %3606 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3607 = stablehlo.divide %3229, %3606 : tensor<576xbf16>
    %cst_527 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3608 = stablehlo.broadcast_in_dim %cst_527, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3609 = stablehlo.multiply %3607, %3608 : tensor<576xbf16>
    %3610 = stablehlo.select %3580, %3229, %3609 : tensor<i1>, tensor<576xbf16>
    %3611 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3612 = stablehlo.divide %3214, %3611 : tensor<576x9x128xbf16>
    %cst_528 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3613 = stablehlo.broadcast_in_dim %cst_528, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3614 = stablehlo.multiply %3612, %3613 : tensor<576x9x128xbf16>
    %3615 = stablehlo.select %3580, %3214, %3614 : tensor<i1>, tensor<576x9x128xbf16>
    %3616 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3617 = stablehlo.divide %3139, %3616 : tensor<9x128x576xbf16>
    %cst_529 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3618 = stablehlo.broadcast_in_dim %cst_529, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3619 = stablehlo.multiply %3617, %3618 : tensor<9x128x576xbf16>
    %3620 = stablehlo.select %3580, %3139, %3619 : tensor<i1>, tensor<9x128x576xbf16>
    %3621 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3622 = stablehlo.divide %3218, %3621 : tensor<576x9x128xbf16>
    %cst_530 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3623 = stablehlo.broadcast_in_dim %cst_530, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3624 = stablehlo.multiply %3622, %3623 : tensor<576x9x128xbf16>
    %3625 = stablehlo.select %3580, %3218, %3624 : tensor<i1>, tensor<576x9x128xbf16>
    %3626 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3627 = stablehlo.divide %3211, %3626 : tensor<576x9x128xbf16>
    %cst_531 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3628 = stablehlo.broadcast_in_dim %cst_531, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3629 = stablehlo.multiply %3627, %3628 : tensor<576x9x128xbf16>
    %3630 = stablehlo.select %3580, %3211, %3629 : tensor<i1>, tensor<576x9x128xbf16>
    %3631 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3632 = stablehlo.divide %2827, %3631 : tensor<576x2016xbf16>
    %cst_532 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3633 = stablehlo.broadcast_in_dim %cst_532, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3634 = stablehlo.multiply %3632, %3633 : tensor<576x2016xbf16>
    %3635 = stablehlo.select %3580, %2827, %3634 : tensor<i1>, tensor<576x2016xbf16>
    %3636 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3637 = stablehlo.divide %2823, %3636 : tensor<576x2016xbf16>
    %cst_533 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3638 = stablehlo.broadcast_in_dim %cst_533, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3639 = stablehlo.multiply %3637, %3638 : tensor<576x2016xbf16>
    %3640 = stablehlo.select %3580, %2823, %3639 : tensor<i1>, tensor<576x2016xbf16>
    %3641 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3642 = stablehlo.divide %2817, %3641 : tensor<2016x576xbf16>
    %cst_534 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3643 = stablehlo.broadcast_in_dim %cst_534, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3644 = stablehlo.multiply %3642, %3643 : tensor<2016x576xbf16>
    %3645 = stablehlo.select %3580, %2817, %3644 : tensor<i1>, tensor<2016x576xbf16>
    %3646 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3647 = stablehlo.divide %2836, %3646 : tensor<576xbf16>
    %cst_535 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3648 = stablehlo.broadcast_in_dim %cst_535, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3649 = stablehlo.multiply %3647, %3648 : tensor<576xbf16>
    %3650 = stablehlo.select %3580, %2836, %3649 : tensor<i1>, tensor<576xbf16>
    %3651 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3652 = stablehlo.divide %2944, %3651 : tensor<576xbf16>
    %cst_536 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3653 = stablehlo.broadcast_in_dim %cst_536, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3654 = stablehlo.multiply %3652, %3653 : tensor<576xbf16>
    %3655 = stablehlo.select %3580, %2944, %3654 : tensor<i1>, tensor<576xbf16>
    %3656 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3657 = stablehlo.divide %2929, %3656 : tensor<576x9x128xbf16>
    %cst_537 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3658 = stablehlo.broadcast_in_dim %cst_537, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3659 = stablehlo.multiply %3657, %3658 : tensor<576x9x128xbf16>
    %3660 = stablehlo.select %3580, %2929, %3659 : tensor<i1>, tensor<576x9x128xbf16>
    %3661 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3662 = stablehlo.divide %2854, %3661 : tensor<9x128x576xbf16>
    %cst_538 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3663 = stablehlo.broadcast_in_dim %cst_538, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3664 = stablehlo.multiply %3662, %3663 : tensor<9x128x576xbf16>
    %3665 = stablehlo.select %3580, %2854, %3664 : tensor<i1>, tensor<9x128x576xbf16>
    %3666 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3667 = stablehlo.divide %2933, %3666 : tensor<576x9x128xbf16>
    %cst_539 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3668 = stablehlo.broadcast_in_dim %cst_539, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3669 = stablehlo.multiply %3667, %3668 : tensor<576x9x128xbf16>
    %3670 = stablehlo.select %3580, %2933, %3669 : tensor<i1>, tensor<576x9x128xbf16>
    %3671 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3672 = stablehlo.divide %2926, %3671 : tensor<576x9x128xbf16>
    %cst_540 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3673 = stablehlo.broadcast_in_dim %cst_540, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3674 = stablehlo.multiply %3672, %3673 : tensor<576x9x128xbf16>
    %3675 = stablehlo.select %3580, %2926, %3674 : tensor<i1>, tensor<576x9x128xbf16>
    %3676 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3677 = stablehlo.divide %2542, %3676 : tensor<576x2016xbf16>
    %cst_541 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3678 = stablehlo.broadcast_in_dim %cst_541, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3679 = stablehlo.multiply %3677, %3678 : tensor<576x2016xbf16>
    %3680 = stablehlo.select %3580, %2542, %3679 : tensor<i1>, tensor<576x2016xbf16>
    %3681 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3682 = stablehlo.divide %2538, %3681 : tensor<576x2016xbf16>
    %cst_542 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3683 = stablehlo.broadcast_in_dim %cst_542, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3684 = stablehlo.multiply %3682, %3683 : tensor<576x2016xbf16>
    %3685 = stablehlo.select %3580, %2538, %3684 : tensor<i1>, tensor<576x2016xbf16>
    %3686 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3687 = stablehlo.divide %2532, %3686 : tensor<2016x576xbf16>
    %cst_543 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3688 = stablehlo.broadcast_in_dim %cst_543, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3689 = stablehlo.multiply %3687, %3688 : tensor<2016x576xbf16>
    %3690 = stablehlo.select %3580, %2532, %3689 : tensor<i1>, tensor<2016x576xbf16>
    %3691 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3692 = stablehlo.divide %2551, %3691 : tensor<576xbf16>
    %cst_544 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3693 = stablehlo.broadcast_in_dim %cst_544, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3694 = stablehlo.multiply %3692, %3693 : tensor<576xbf16>
    %3695 = stablehlo.select %3580, %2551, %3694 : tensor<i1>, tensor<576xbf16>
    %3696 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3697 = stablehlo.divide %2659, %3696 : tensor<576xbf16>
    %cst_545 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3698 = stablehlo.broadcast_in_dim %cst_545, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3699 = stablehlo.multiply %3697, %3698 : tensor<576xbf16>
    %3700 = stablehlo.select %3580, %2659, %3699 : tensor<i1>, tensor<576xbf16>
    %3701 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3702 = stablehlo.divide %2644, %3701 : tensor<576x9x128xbf16>
    %cst_546 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3703 = stablehlo.broadcast_in_dim %cst_546, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3704 = stablehlo.multiply %3702, %3703 : tensor<576x9x128xbf16>
    %3705 = stablehlo.select %3580, %2644, %3704 : tensor<i1>, tensor<576x9x128xbf16>
    %3706 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3707 = stablehlo.divide %2569, %3706 : tensor<9x128x576xbf16>
    %cst_547 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3708 = stablehlo.broadcast_in_dim %cst_547, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3709 = stablehlo.multiply %3707, %3708 : tensor<9x128x576xbf16>
    %3710 = stablehlo.select %3580, %2569, %3709 : tensor<i1>, tensor<9x128x576xbf16>
    %3711 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3712 = stablehlo.divide %2648, %3711 : tensor<576x9x128xbf16>
    %cst_548 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3713 = stablehlo.broadcast_in_dim %cst_548, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3714 = stablehlo.multiply %3712, %3713 : tensor<576x9x128xbf16>
    %3715 = stablehlo.select %3580, %2648, %3714 : tensor<i1>, tensor<576x9x128xbf16>
    %3716 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3717 = stablehlo.divide %2641, %3716 : tensor<576x9x128xbf16>
    %cst_549 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3718 = stablehlo.broadcast_in_dim %cst_549, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3719 = stablehlo.multiply %3717, %3718 : tensor<576x9x128xbf16>
    %3720 = stablehlo.select %3580, %2641, %3719 : tensor<i1>, tensor<576x9x128xbf16>
    %3721 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3722 = stablehlo.divide %2257, %3721 : tensor<576x2016xbf16>
    %cst_550 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3723 = stablehlo.broadcast_in_dim %cst_550, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3724 = stablehlo.multiply %3722, %3723 : tensor<576x2016xbf16>
    %3725 = stablehlo.select %3580, %2257, %3724 : tensor<i1>, tensor<576x2016xbf16>
    %3726 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3727 = stablehlo.divide %2253, %3726 : tensor<576x2016xbf16>
    %cst_551 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3728 = stablehlo.broadcast_in_dim %cst_551, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3729 = stablehlo.multiply %3727, %3728 : tensor<576x2016xbf16>
    %3730 = stablehlo.select %3580, %2253, %3729 : tensor<i1>, tensor<576x2016xbf16>
    %3731 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3732 = stablehlo.divide %2247, %3731 : tensor<2016x576xbf16>
    %cst_552 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3733 = stablehlo.broadcast_in_dim %cst_552, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3734 = stablehlo.multiply %3732, %3733 : tensor<2016x576xbf16>
    %3735 = stablehlo.select %3580, %2247, %3734 : tensor<i1>, tensor<2016x576xbf16>
    %3736 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3737 = stablehlo.divide %2266, %3736 : tensor<576xbf16>
    %cst_553 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3738 = stablehlo.broadcast_in_dim %cst_553, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3739 = stablehlo.multiply %3737, %3738 : tensor<576xbf16>
    %3740 = stablehlo.select %3580, %2266, %3739 : tensor<i1>, tensor<576xbf16>
    %3741 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3742 = stablehlo.divide %2374, %3741 : tensor<576xbf16>
    %cst_554 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3743 = stablehlo.broadcast_in_dim %cst_554, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3744 = stablehlo.multiply %3742, %3743 : tensor<576xbf16>
    %3745 = stablehlo.select %3580, %2374, %3744 : tensor<i1>, tensor<576xbf16>
    %3746 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3747 = stablehlo.divide %2359, %3746 : tensor<576x9x128xbf16>
    %cst_555 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3748 = stablehlo.broadcast_in_dim %cst_555, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3749 = stablehlo.multiply %3747, %3748 : tensor<576x9x128xbf16>
    %3750 = stablehlo.select %3580, %2359, %3749 : tensor<i1>, tensor<576x9x128xbf16>
    %3751 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3752 = stablehlo.divide %2284, %3751 : tensor<9x128x576xbf16>
    %cst_556 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3753 = stablehlo.broadcast_in_dim %cst_556, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3754 = stablehlo.multiply %3752, %3753 : tensor<9x128x576xbf16>
    %3755 = stablehlo.select %3580, %2284, %3754 : tensor<i1>, tensor<9x128x576xbf16>
    %3756 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3757 = stablehlo.divide %2363, %3756 : tensor<576x9x128xbf16>
    %cst_557 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3758 = stablehlo.broadcast_in_dim %cst_557, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3759 = stablehlo.multiply %3757, %3758 : tensor<576x9x128xbf16>
    %3760 = stablehlo.select %3580, %2363, %3759 : tensor<i1>, tensor<576x9x128xbf16>
    %3761 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3762 = stablehlo.divide %2356, %3761 : tensor<576x9x128xbf16>
    %cst_558 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3763 = stablehlo.broadcast_in_dim %cst_558, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3764 = stablehlo.multiply %3762, %3763 : tensor<576x9x128xbf16>
    %3765 = stablehlo.select %3580, %2356, %3764 : tensor<i1>, tensor<576x9x128xbf16>
    %3766 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3767 = stablehlo.divide %1972, %3766 : tensor<576x2016xbf16>
    %cst_559 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3768 = stablehlo.broadcast_in_dim %cst_559, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3769 = stablehlo.multiply %3767, %3768 : tensor<576x2016xbf16>
    %3770 = stablehlo.select %3580, %1972, %3769 : tensor<i1>, tensor<576x2016xbf16>
    %3771 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3772 = stablehlo.divide %1968, %3771 : tensor<576x2016xbf16>
    %cst_560 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3773 = stablehlo.broadcast_in_dim %cst_560, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3774 = stablehlo.multiply %3772, %3773 : tensor<576x2016xbf16>
    %3775 = stablehlo.select %3580, %1968, %3774 : tensor<i1>, tensor<576x2016xbf16>
    %3776 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3777 = stablehlo.divide %1962, %3776 : tensor<2016x576xbf16>
    %cst_561 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3778 = stablehlo.broadcast_in_dim %cst_561, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3779 = stablehlo.multiply %3777, %3778 : tensor<2016x576xbf16>
    %3780 = stablehlo.select %3580, %1962, %3779 : tensor<i1>, tensor<2016x576xbf16>
    %3781 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3782 = stablehlo.divide %1981, %3781 : tensor<576xbf16>
    %cst_562 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3783 = stablehlo.broadcast_in_dim %cst_562, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3784 = stablehlo.multiply %3782, %3783 : tensor<576xbf16>
    %3785 = stablehlo.select %3580, %1981, %3784 : tensor<i1>, tensor<576xbf16>
    %3786 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3787 = stablehlo.divide %2089, %3786 : tensor<576xbf16>
    %cst_563 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3788 = stablehlo.broadcast_in_dim %cst_563, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3789 = stablehlo.multiply %3787, %3788 : tensor<576xbf16>
    %3790 = stablehlo.select %3580, %2089, %3789 : tensor<i1>, tensor<576xbf16>
    %3791 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3792 = stablehlo.divide %2074, %3791 : tensor<576x9x128xbf16>
    %cst_564 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3793 = stablehlo.broadcast_in_dim %cst_564, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3794 = stablehlo.multiply %3792, %3793 : tensor<576x9x128xbf16>
    %3795 = stablehlo.select %3580, %2074, %3794 : tensor<i1>, tensor<576x9x128xbf16>
    %3796 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3797 = stablehlo.divide %1999, %3796 : tensor<9x128x576xbf16>
    %cst_565 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3798 = stablehlo.broadcast_in_dim %cst_565, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3799 = stablehlo.multiply %3797, %3798 : tensor<9x128x576xbf16>
    %3800 = stablehlo.select %3580, %1999, %3799 : tensor<i1>, tensor<9x128x576xbf16>
    %3801 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3802 = stablehlo.divide %2078, %3801 : tensor<576x9x128xbf16>
    %cst_566 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3803 = stablehlo.broadcast_in_dim %cst_566, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3804 = stablehlo.multiply %3802, %3803 : tensor<576x9x128xbf16>
    %3805 = stablehlo.select %3580, %2078, %3804 : tensor<i1>, tensor<576x9x128xbf16>
    %3806 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3807 = stablehlo.divide %2071, %3806 : tensor<576x9x128xbf16>
    %cst_567 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3808 = stablehlo.broadcast_in_dim %cst_567, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3809 = stablehlo.multiply %3807, %3808 : tensor<576x9x128xbf16>
    %3810 = stablehlo.select %3580, %2071, %3809 : tensor<i1>, tensor<576x9x128xbf16>
    %3811 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3812 = stablehlo.divide %1687, %3811 : tensor<576x2016xbf16>
    %cst_568 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3813 = stablehlo.broadcast_in_dim %cst_568, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3814 = stablehlo.multiply %3812, %3813 : tensor<576x2016xbf16>
    %3815 = stablehlo.select %3580, %1687, %3814 : tensor<i1>, tensor<576x2016xbf16>
    %3816 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3817 = stablehlo.divide %1683, %3816 : tensor<576x2016xbf16>
    %cst_569 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3818 = stablehlo.broadcast_in_dim %cst_569, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3819 = stablehlo.multiply %3817, %3818 : tensor<576x2016xbf16>
    %3820 = stablehlo.select %3580, %1683, %3819 : tensor<i1>, tensor<576x2016xbf16>
    %3821 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3822 = stablehlo.divide %1677, %3821 : tensor<2016x576xbf16>
    %cst_570 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3823 = stablehlo.broadcast_in_dim %cst_570, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3824 = stablehlo.multiply %3822, %3823 : tensor<2016x576xbf16>
    %3825 = stablehlo.select %3580, %1677, %3824 : tensor<i1>, tensor<2016x576xbf16>
    %3826 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3827 = stablehlo.divide %1696, %3826 : tensor<576xbf16>
    %cst_571 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3828 = stablehlo.broadcast_in_dim %cst_571, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3829 = stablehlo.multiply %3827, %3828 : tensor<576xbf16>
    %3830 = stablehlo.select %3580, %1696, %3829 : tensor<i1>, tensor<576xbf16>
    %3831 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3832 = stablehlo.divide %1804, %3831 : tensor<576xbf16>
    %cst_572 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3833 = stablehlo.broadcast_in_dim %cst_572, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3834 = stablehlo.multiply %3832, %3833 : tensor<576xbf16>
    %3835 = stablehlo.select %3580, %1804, %3834 : tensor<i1>, tensor<576xbf16>
    %3836 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3837 = stablehlo.divide %1789, %3836 : tensor<576x9x128xbf16>
    %cst_573 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3838 = stablehlo.broadcast_in_dim %cst_573, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3839 = stablehlo.multiply %3837, %3838 : tensor<576x9x128xbf16>
    %3840 = stablehlo.select %3580, %1789, %3839 : tensor<i1>, tensor<576x9x128xbf16>
    %3841 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3842 = stablehlo.divide %1714, %3841 : tensor<9x128x576xbf16>
    %cst_574 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3843 = stablehlo.broadcast_in_dim %cst_574, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3844 = stablehlo.multiply %3842, %3843 : tensor<9x128x576xbf16>
    %3845 = stablehlo.select %3580, %1714, %3844 : tensor<i1>, tensor<9x128x576xbf16>
    %3846 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3847 = stablehlo.divide %1793, %3846 : tensor<576x9x128xbf16>
    %cst_575 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3848 = stablehlo.broadcast_in_dim %cst_575, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3849 = stablehlo.multiply %3847, %3848 : tensor<576x9x128xbf16>
    %3850 = stablehlo.select %3580, %1793, %3849 : tensor<i1>, tensor<576x9x128xbf16>
    %3851 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3852 = stablehlo.divide %1786, %3851 : tensor<576x9x128xbf16>
    %cst_576 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3853 = stablehlo.broadcast_in_dim %cst_576, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3854 = stablehlo.multiply %3852, %3853 : tensor<576x9x128xbf16>
    %3855 = stablehlo.select %3580, %1786, %3854 : tensor<i1>, tensor<576x9x128xbf16>
    %3856 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3857 = stablehlo.divide %1402, %3856 : tensor<576x2016xbf16>
    %cst_577 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3858 = stablehlo.broadcast_in_dim %cst_577, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3859 = stablehlo.multiply %3857, %3858 : tensor<576x2016xbf16>
    %3860 = stablehlo.select %3580, %1402, %3859 : tensor<i1>, tensor<576x2016xbf16>
    %3861 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3862 = stablehlo.divide %1398, %3861 : tensor<576x2016xbf16>
    %cst_578 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3863 = stablehlo.broadcast_in_dim %cst_578, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3864 = stablehlo.multiply %3862, %3863 : tensor<576x2016xbf16>
    %3865 = stablehlo.select %3580, %1398, %3864 : tensor<i1>, tensor<576x2016xbf16>
    %3866 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3867 = stablehlo.divide %1392, %3866 : tensor<2016x576xbf16>
    %cst_579 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3868 = stablehlo.broadcast_in_dim %cst_579, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3869 = stablehlo.multiply %3867, %3868 : tensor<2016x576xbf16>
    %3870 = stablehlo.select %3580, %1392, %3869 : tensor<i1>, tensor<2016x576xbf16>
    %3871 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3872 = stablehlo.divide %1411, %3871 : tensor<576xbf16>
    %cst_580 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3873 = stablehlo.broadcast_in_dim %cst_580, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3874 = stablehlo.multiply %3872, %3873 : tensor<576xbf16>
    %3875 = stablehlo.select %3580, %1411, %3874 : tensor<i1>, tensor<576xbf16>
    %3876 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3877 = stablehlo.divide %1519, %3876 : tensor<576xbf16>
    %cst_581 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3878 = stablehlo.broadcast_in_dim %cst_581, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3879 = stablehlo.multiply %3877, %3878 : tensor<576xbf16>
    %3880 = stablehlo.select %3580, %1519, %3879 : tensor<i1>, tensor<576xbf16>
    %3881 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3882 = stablehlo.divide %1504, %3881 : tensor<576x9x128xbf16>
    %cst_582 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3883 = stablehlo.broadcast_in_dim %cst_582, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3884 = stablehlo.multiply %3882, %3883 : tensor<576x9x128xbf16>
    %3885 = stablehlo.select %3580, %1504, %3884 : tensor<i1>, tensor<576x9x128xbf16>
    %3886 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3887 = stablehlo.divide %1429, %3886 : tensor<9x128x576xbf16>
    %cst_583 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3888 = stablehlo.broadcast_in_dim %cst_583, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3889 = stablehlo.multiply %3887, %3888 : tensor<9x128x576xbf16>
    %3890 = stablehlo.select %3580, %1429, %3889 : tensor<i1>, tensor<9x128x576xbf16>
    %3891 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3892 = stablehlo.divide %1508, %3891 : tensor<576x9x128xbf16>
    %cst_584 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3893 = stablehlo.broadcast_in_dim %cst_584, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3894 = stablehlo.multiply %3892, %3893 : tensor<576x9x128xbf16>
    %3895 = stablehlo.select %3580, %1508, %3894 : tensor<i1>, tensor<576x9x128xbf16>
    %3896 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3897 = stablehlo.divide %1501, %3896 : tensor<576x9x128xbf16>
    %cst_585 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3898 = stablehlo.broadcast_in_dim %cst_585, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3899 = stablehlo.multiply %3897, %3898 : tensor<576x9x128xbf16>
    %3900 = stablehlo.select %3580, %1501, %3899 : tensor<i1>, tensor<576x9x128xbf16>
    %3901 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<576x128256xbf16>
    %3902 = stablehlo.divide %1229, %3901 : tensor<576x128256xbf16>
    %cst_586 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3903 = stablehlo.broadcast_in_dim %cst_586, dims = [] : (tensor<bf16>) -> tensor<576x128256xbf16>
    %3904 = stablehlo.multiply %3902, %3903 : tensor<576x128256xbf16>
    %3905 = stablehlo.select %3580, %1229, %3904 : tensor<i1>, tensor<576x128256xbf16>
    %3906 = stablehlo.broadcast_in_dim %3579, dims = [] : (tensor<bf16>) -> tensor<128256x576xbf16>
    %3907 = stablehlo.divide %3248, %3906 : tensor<128256x576xbf16>
    %cst_587 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3908 = stablehlo.broadcast_in_dim %cst_587, dims = [] : (tensor<bf16>) -> tensor<128256x576xbf16>
    %3909 = stablehlo.multiply %3907, %3908 : tensor<128256x576xbf16>
    %3910 = stablehlo.select %3580, %3248, %3909 : tensor<i1>, tensor<128256x576xbf16>
    %cst_588 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3911 = stablehlo.broadcast_in_dim %cst_588, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3912 = stablehlo.multiply %3911, %3585 : tensor<576xbf16>
    %cst_589 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3913 = stablehlo.broadcast_in_dim %cst_589, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3914 = stablehlo.multiply %3913, %arg68 : tensor<576xbf16>
    %3915 = stablehlo.add %3912, %3914 : tensor<576xbf16>
    %cst_590 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3916 = stablehlo.broadcast_in_dim %cst_590, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3917 = stablehlo.multiply %3916, %3590 : tensor<576x2016xbf16>
    %cst_591 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3918 = stablehlo.broadcast_in_dim %cst_591, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3919 = stablehlo.multiply %3918, %arg69 : tensor<576x2016xbf16>
    %3920 = stablehlo.add %3917, %3919 : tensor<576x2016xbf16>
    %cst_592 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3921 = stablehlo.broadcast_in_dim %cst_592, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3922 = stablehlo.multiply %3921, %3595 : tensor<576x2016xbf16>
    %cst_593 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3923 = stablehlo.broadcast_in_dim %cst_593, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3924 = stablehlo.multiply %3923, %arg70 : tensor<576x2016xbf16>
    %3925 = stablehlo.add %3922, %3924 : tensor<576x2016xbf16>
    %cst_594 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3926 = stablehlo.broadcast_in_dim %cst_594, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3927 = stablehlo.multiply %3926, %3600 : tensor<2016x576xbf16>
    %cst_595 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3928 = stablehlo.broadcast_in_dim %cst_595, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3929 = stablehlo.multiply %3928, %arg71 : tensor<2016x576xbf16>
    %3930 = stablehlo.add %3927, %3929 : tensor<2016x576xbf16>
    %cst_596 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3931 = stablehlo.broadcast_in_dim %cst_596, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3932 = stablehlo.multiply %3931, %3605 : tensor<576xbf16>
    %cst_597 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3933 = stablehlo.broadcast_in_dim %cst_597, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3934 = stablehlo.multiply %3933, %arg72 : tensor<576xbf16>
    %3935 = stablehlo.add %3932, %3934 : tensor<576xbf16>
    %cst_598 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3936 = stablehlo.broadcast_in_dim %cst_598, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3937 = stablehlo.multiply %3936, %3610 : tensor<576xbf16>
    %cst_599 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3938 = stablehlo.broadcast_in_dim %cst_599, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3939 = stablehlo.multiply %3938, %arg73 : tensor<576xbf16>
    %3940 = stablehlo.add %3937, %3939 : tensor<576xbf16>
    %cst_600 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3941 = stablehlo.broadcast_in_dim %cst_600, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3942 = stablehlo.multiply %3941, %3615 : tensor<576x9x128xbf16>
    %cst_601 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3943 = stablehlo.broadcast_in_dim %cst_601, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3944 = stablehlo.multiply %3943, %arg74 : tensor<576x9x128xbf16>
    %3945 = stablehlo.add %3942, %3944 : tensor<576x9x128xbf16>
    %cst_602 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3946 = stablehlo.broadcast_in_dim %cst_602, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3947 = stablehlo.multiply %3946, %3620 : tensor<9x128x576xbf16>
    %cst_603 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3948 = stablehlo.broadcast_in_dim %cst_603, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3949 = stablehlo.multiply %3948, %arg75 : tensor<9x128x576xbf16>
    %3950 = stablehlo.add %3947, %3949 : tensor<9x128x576xbf16>
    %cst_604 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3951 = stablehlo.broadcast_in_dim %cst_604, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3952 = stablehlo.multiply %3951, %3625 : tensor<576x9x128xbf16>
    %cst_605 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3953 = stablehlo.broadcast_in_dim %cst_605, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3954 = stablehlo.multiply %3953, %arg76 : tensor<576x9x128xbf16>
    %3955 = stablehlo.add %3952, %3954 : tensor<576x9x128xbf16>
    %cst_606 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3956 = stablehlo.broadcast_in_dim %cst_606, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3957 = stablehlo.multiply %3956, %3630 : tensor<576x9x128xbf16>
    %cst_607 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3958 = stablehlo.broadcast_in_dim %cst_607, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3959 = stablehlo.multiply %3958, %arg77 : tensor<576x9x128xbf16>
    %3960 = stablehlo.add %3957, %3959 : tensor<576x9x128xbf16>
    %cst_608 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3961 = stablehlo.broadcast_in_dim %cst_608, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3962 = stablehlo.multiply %3961, %3635 : tensor<576x2016xbf16>
    %cst_609 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3963 = stablehlo.broadcast_in_dim %cst_609, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3964 = stablehlo.multiply %3963, %arg78 : tensor<576x2016xbf16>
    %3965 = stablehlo.add %3962, %3964 : tensor<576x2016xbf16>
    %cst_610 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3966 = stablehlo.broadcast_in_dim %cst_610, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3967 = stablehlo.multiply %3966, %3640 : tensor<576x2016xbf16>
    %cst_611 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3968 = stablehlo.broadcast_in_dim %cst_611, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %3969 = stablehlo.multiply %3968, %arg79 : tensor<576x2016xbf16>
    %3970 = stablehlo.add %3967, %3969 : tensor<576x2016xbf16>
    %cst_612 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3971 = stablehlo.broadcast_in_dim %cst_612, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3972 = stablehlo.multiply %3971, %3645 : tensor<2016x576xbf16>
    %cst_613 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3973 = stablehlo.broadcast_in_dim %cst_613, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %3974 = stablehlo.multiply %3973, %arg80 : tensor<2016x576xbf16>
    %3975 = stablehlo.add %3972, %3974 : tensor<2016x576xbf16>
    %cst_614 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3976 = stablehlo.broadcast_in_dim %cst_614, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3977 = stablehlo.multiply %3976, %3650 : tensor<576xbf16>
    %cst_615 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3978 = stablehlo.broadcast_in_dim %cst_615, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3979 = stablehlo.multiply %3978, %arg81 : tensor<576xbf16>
    %3980 = stablehlo.add %3977, %3979 : tensor<576xbf16>
    %cst_616 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3981 = stablehlo.broadcast_in_dim %cst_616, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3982 = stablehlo.multiply %3981, %3655 : tensor<576xbf16>
    %cst_617 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3983 = stablehlo.broadcast_in_dim %cst_617, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %3984 = stablehlo.multiply %3983, %arg82 : tensor<576xbf16>
    %3985 = stablehlo.add %3982, %3984 : tensor<576xbf16>
    %cst_618 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3986 = stablehlo.broadcast_in_dim %cst_618, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3987 = stablehlo.multiply %3986, %3660 : tensor<576x9x128xbf16>
    %cst_619 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3988 = stablehlo.broadcast_in_dim %cst_619, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3989 = stablehlo.multiply %3988, %arg83 : tensor<576x9x128xbf16>
    %3990 = stablehlo.add %3987, %3989 : tensor<576x9x128xbf16>
    %cst_620 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3991 = stablehlo.broadcast_in_dim %cst_620, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3992 = stablehlo.multiply %3991, %3665 : tensor<9x128x576xbf16>
    %cst_621 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3993 = stablehlo.broadcast_in_dim %cst_621, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %3994 = stablehlo.multiply %3993, %arg84 : tensor<9x128x576xbf16>
    %3995 = stablehlo.add %3992, %3994 : tensor<9x128x576xbf16>
    %cst_622 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %3996 = stablehlo.broadcast_in_dim %cst_622, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3997 = stablehlo.multiply %3996, %3670 : tensor<576x9x128xbf16>
    %cst_623 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %3998 = stablehlo.broadcast_in_dim %cst_623, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %3999 = stablehlo.multiply %3998, %arg85 : tensor<576x9x128xbf16>
    %4000 = stablehlo.add %3997, %3999 : tensor<576x9x128xbf16>
    %cst_624 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4001 = stablehlo.broadcast_in_dim %cst_624, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4002 = stablehlo.multiply %4001, %3675 : tensor<576x9x128xbf16>
    %cst_625 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4003 = stablehlo.broadcast_in_dim %cst_625, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4004 = stablehlo.multiply %4003, %arg86 : tensor<576x9x128xbf16>
    %4005 = stablehlo.add %4002, %4004 : tensor<576x9x128xbf16>
    %cst_626 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4006 = stablehlo.broadcast_in_dim %cst_626, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4007 = stablehlo.multiply %4006, %3680 : tensor<576x2016xbf16>
    %cst_627 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4008 = stablehlo.broadcast_in_dim %cst_627, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4009 = stablehlo.multiply %4008, %arg87 : tensor<576x2016xbf16>
    %4010 = stablehlo.add %4007, %4009 : tensor<576x2016xbf16>
    %cst_628 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4011 = stablehlo.broadcast_in_dim %cst_628, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4012 = stablehlo.multiply %4011, %3685 : tensor<576x2016xbf16>
    %cst_629 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4013 = stablehlo.broadcast_in_dim %cst_629, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4014 = stablehlo.multiply %4013, %arg88 : tensor<576x2016xbf16>
    %4015 = stablehlo.add %4012, %4014 : tensor<576x2016xbf16>
    %cst_630 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4016 = stablehlo.broadcast_in_dim %cst_630, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4017 = stablehlo.multiply %4016, %3690 : tensor<2016x576xbf16>
    %cst_631 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4018 = stablehlo.broadcast_in_dim %cst_631, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4019 = stablehlo.multiply %4018, %arg89 : tensor<2016x576xbf16>
    %4020 = stablehlo.add %4017, %4019 : tensor<2016x576xbf16>
    %cst_632 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4021 = stablehlo.broadcast_in_dim %cst_632, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4022 = stablehlo.multiply %4021, %3695 : tensor<576xbf16>
    %cst_633 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4023 = stablehlo.broadcast_in_dim %cst_633, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4024 = stablehlo.multiply %4023, %arg90 : tensor<576xbf16>
    %4025 = stablehlo.add %4022, %4024 : tensor<576xbf16>
    %cst_634 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4026 = stablehlo.broadcast_in_dim %cst_634, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4027 = stablehlo.multiply %4026, %3700 : tensor<576xbf16>
    %cst_635 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4028 = stablehlo.broadcast_in_dim %cst_635, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4029 = stablehlo.multiply %4028, %arg91 : tensor<576xbf16>
    %4030 = stablehlo.add %4027, %4029 : tensor<576xbf16>
    %cst_636 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4031 = stablehlo.broadcast_in_dim %cst_636, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4032 = stablehlo.multiply %4031, %3705 : tensor<576x9x128xbf16>
    %cst_637 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4033 = stablehlo.broadcast_in_dim %cst_637, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4034 = stablehlo.multiply %4033, %arg92 : tensor<576x9x128xbf16>
    %4035 = stablehlo.add %4032, %4034 : tensor<576x9x128xbf16>
    %cst_638 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4036 = stablehlo.broadcast_in_dim %cst_638, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4037 = stablehlo.multiply %4036, %3710 : tensor<9x128x576xbf16>
    %cst_639 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4038 = stablehlo.broadcast_in_dim %cst_639, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4039 = stablehlo.multiply %4038, %arg93 : tensor<9x128x576xbf16>
    %4040 = stablehlo.add %4037, %4039 : tensor<9x128x576xbf16>
    %cst_640 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4041 = stablehlo.broadcast_in_dim %cst_640, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4042 = stablehlo.multiply %4041, %3715 : tensor<576x9x128xbf16>
    %cst_641 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4043 = stablehlo.broadcast_in_dim %cst_641, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4044 = stablehlo.multiply %4043, %arg94 : tensor<576x9x128xbf16>
    %4045 = stablehlo.add %4042, %4044 : tensor<576x9x128xbf16>
    %cst_642 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4046 = stablehlo.broadcast_in_dim %cst_642, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4047 = stablehlo.multiply %4046, %3720 : tensor<576x9x128xbf16>
    %cst_643 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4048 = stablehlo.broadcast_in_dim %cst_643, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4049 = stablehlo.multiply %4048, %arg95 : tensor<576x9x128xbf16>
    %4050 = stablehlo.add %4047, %4049 : tensor<576x9x128xbf16>
    %cst_644 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4051 = stablehlo.broadcast_in_dim %cst_644, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4052 = stablehlo.multiply %4051, %3725 : tensor<576x2016xbf16>
    %cst_645 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4053 = stablehlo.broadcast_in_dim %cst_645, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4054 = stablehlo.multiply %4053, %arg96 : tensor<576x2016xbf16>
    %4055 = stablehlo.add %4052, %4054 : tensor<576x2016xbf16>
    %cst_646 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4056 = stablehlo.broadcast_in_dim %cst_646, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4057 = stablehlo.multiply %4056, %3730 : tensor<576x2016xbf16>
    %cst_647 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4058 = stablehlo.broadcast_in_dim %cst_647, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4059 = stablehlo.multiply %4058, %arg97 : tensor<576x2016xbf16>
    %4060 = stablehlo.add %4057, %4059 : tensor<576x2016xbf16>
    %cst_648 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4061 = stablehlo.broadcast_in_dim %cst_648, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4062 = stablehlo.multiply %4061, %3735 : tensor<2016x576xbf16>
    %cst_649 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4063 = stablehlo.broadcast_in_dim %cst_649, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4064 = stablehlo.multiply %4063, %arg98 : tensor<2016x576xbf16>
    %4065 = stablehlo.add %4062, %4064 : tensor<2016x576xbf16>
    %cst_650 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4066 = stablehlo.broadcast_in_dim %cst_650, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4067 = stablehlo.multiply %4066, %3740 : tensor<576xbf16>
    %cst_651 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4068 = stablehlo.broadcast_in_dim %cst_651, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4069 = stablehlo.multiply %4068, %arg99 : tensor<576xbf16>
    %4070 = stablehlo.add %4067, %4069 : tensor<576xbf16>
    %cst_652 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4071 = stablehlo.broadcast_in_dim %cst_652, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4072 = stablehlo.multiply %4071, %3745 : tensor<576xbf16>
    %cst_653 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4073 = stablehlo.broadcast_in_dim %cst_653, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4074 = stablehlo.multiply %4073, %arg100 : tensor<576xbf16>
    %4075 = stablehlo.add %4072, %4074 : tensor<576xbf16>
    %cst_654 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4076 = stablehlo.broadcast_in_dim %cst_654, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4077 = stablehlo.multiply %4076, %3750 : tensor<576x9x128xbf16>
    %cst_655 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4078 = stablehlo.broadcast_in_dim %cst_655, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4079 = stablehlo.multiply %4078, %arg101 : tensor<576x9x128xbf16>
    %4080 = stablehlo.add %4077, %4079 : tensor<576x9x128xbf16>
    %cst_656 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4081 = stablehlo.broadcast_in_dim %cst_656, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4082 = stablehlo.multiply %4081, %3755 : tensor<9x128x576xbf16>
    %cst_657 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4083 = stablehlo.broadcast_in_dim %cst_657, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4084 = stablehlo.multiply %4083, %arg102 : tensor<9x128x576xbf16>
    %4085 = stablehlo.add %4082, %4084 : tensor<9x128x576xbf16>
    %cst_658 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4086 = stablehlo.broadcast_in_dim %cst_658, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4087 = stablehlo.multiply %4086, %3760 : tensor<576x9x128xbf16>
    %cst_659 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4088 = stablehlo.broadcast_in_dim %cst_659, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4089 = stablehlo.multiply %4088, %arg103 : tensor<576x9x128xbf16>
    %4090 = stablehlo.add %4087, %4089 : tensor<576x9x128xbf16>
    %cst_660 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4091 = stablehlo.broadcast_in_dim %cst_660, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4092 = stablehlo.multiply %4091, %3765 : tensor<576x9x128xbf16>
    %cst_661 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4093 = stablehlo.broadcast_in_dim %cst_661, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4094 = stablehlo.multiply %4093, %arg104 : tensor<576x9x128xbf16>
    %4095 = stablehlo.add %4092, %4094 : tensor<576x9x128xbf16>
    %cst_662 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4096 = stablehlo.broadcast_in_dim %cst_662, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4097 = stablehlo.multiply %4096, %3770 : tensor<576x2016xbf16>
    %cst_663 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4098 = stablehlo.broadcast_in_dim %cst_663, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4099 = stablehlo.multiply %4098, %arg105 : tensor<576x2016xbf16>
    %4100 = stablehlo.add %4097, %4099 : tensor<576x2016xbf16>
    %cst_664 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4101 = stablehlo.broadcast_in_dim %cst_664, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4102 = stablehlo.multiply %4101, %3775 : tensor<576x2016xbf16>
    %cst_665 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4103 = stablehlo.broadcast_in_dim %cst_665, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4104 = stablehlo.multiply %4103, %arg106 : tensor<576x2016xbf16>
    %4105 = stablehlo.add %4102, %4104 : tensor<576x2016xbf16>
    %cst_666 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4106 = stablehlo.broadcast_in_dim %cst_666, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4107 = stablehlo.multiply %4106, %3780 : tensor<2016x576xbf16>
    %cst_667 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4108 = stablehlo.broadcast_in_dim %cst_667, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4109 = stablehlo.multiply %4108, %arg107 : tensor<2016x576xbf16>
    %4110 = stablehlo.add %4107, %4109 : tensor<2016x576xbf16>
    %cst_668 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4111 = stablehlo.broadcast_in_dim %cst_668, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4112 = stablehlo.multiply %4111, %3785 : tensor<576xbf16>
    %cst_669 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4113 = stablehlo.broadcast_in_dim %cst_669, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4114 = stablehlo.multiply %4113, %arg108 : tensor<576xbf16>
    %4115 = stablehlo.add %4112, %4114 : tensor<576xbf16>
    %cst_670 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4116 = stablehlo.broadcast_in_dim %cst_670, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4117 = stablehlo.multiply %4116, %3790 : tensor<576xbf16>
    %cst_671 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4118 = stablehlo.broadcast_in_dim %cst_671, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4119 = stablehlo.multiply %4118, %arg109 : tensor<576xbf16>
    %4120 = stablehlo.add %4117, %4119 : tensor<576xbf16>
    %cst_672 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4121 = stablehlo.broadcast_in_dim %cst_672, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4122 = stablehlo.multiply %4121, %3795 : tensor<576x9x128xbf16>
    %cst_673 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4123 = stablehlo.broadcast_in_dim %cst_673, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4124 = stablehlo.multiply %4123, %arg110 : tensor<576x9x128xbf16>
    %4125 = stablehlo.add %4122, %4124 : tensor<576x9x128xbf16>
    %cst_674 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4126 = stablehlo.broadcast_in_dim %cst_674, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4127 = stablehlo.multiply %4126, %3800 : tensor<9x128x576xbf16>
    %cst_675 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4128 = stablehlo.broadcast_in_dim %cst_675, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4129 = stablehlo.multiply %4128, %arg111 : tensor<9x128x576xbf16>
    %4130 = stablehlo.add %4127, %4129 : tensor<9x128x576xbf16>
    %cst_676 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4131 = stablehlo.broadcast_in_dim %cst_676, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4132 = stablehlo.multiply %4131, %3805 : tensor<576x9x128xbf16>
    %cst_677 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4133 = stablehlo.broadcast_in_dim %cst_677, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4134 = stablehlo.multiply %4133, %arg112 : tensor<576x9x128xbf16>
    %4135 = stablehlo.add %4132, %4134 : tensor<576x9x128xbf16>
    %cst_678 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4136 = stablehlo.broadcast_in_dim %cst_678, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4137 = stablehlo.multiply %4136, %3810 : tensor<576x9x128xbf16>
    %cst_679 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4138 = stablehlo.broadcast_in_dim %cst_679, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4139 = stablehlo.multiply %4138, %arg113 : tensor<576x9x128xbf16>
    %4140 = stablehlo.add %4137, %4139 : tensor<576x9x128xbf16>
    %cst_680 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4141 = stablehlo.broadcast_in_dim %cst_680, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4142 = stablehlo.multiply %4141, %3815 : tensor<576x2016xbf16>
    %cst_681 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4143 = stablehlo.broadcast_in_dim %cst_681, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4144 = stablehlo.multiply %4143, %arg114 : tensor<576x2016xbf16>
    %4145 = stablehlo.add %4142, %4144 : tensor<576x2016xbf16>
    %cst_682 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4146 = stablehlo.broadcast_in_dim %cst_682, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4147 = stablehlo.multiply %4146, %3820 : tensor<576x2016xbf16>
    %cst_683 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4148 = stablehlo.broadcast_in_dim %cst_683, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4149 = stablehlo.multiply %4148, %arg115 : tensor<576x2016xbf16>
    %4150 = stablehlo.add %4147, %4149 : tensor<576x2016xbf16>
    %cst_684 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4151 = stablehlo.broadcast_in_dim %cst_684, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4152 = stablehlo.multiply %4151, %3825 : tensor<2016x576xbf16>
    %cst_685 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4153 = stablehlo.broadcast_in_dim %cst_685, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4154 = stablehlo.multiply %4153, %arg116 : tensor<2016x576xbf16>
    %4155 = stablehlo.add %4152, %4154 : tensor<2016x576xbf16>
    %cst_686 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4156 = stablehlo.broadcast_in_dim %cst_686, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4157 = stablehlo.multiply %4156, %3830 : tensor<576xbf16>
    %cst_687 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4158 = stablehlo.broadcast_in_dim %cst_687, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4159 = stablehlo.multiply %4158, %arg117 : tensor<576xbf16>
    %4160 = stablehlo.add %4157, %4159 : tensor<576xbf16>
    %cst_688 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4161 = stablehlo.broadcast_in_dim %cst_688, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4162 = stablehlo.multiply %4161, %3835 : tensor<576xbf16>
    %cst_689 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4163 = stablehlo.broadcast_in_dim %cst_689, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4164 = stablehlo.multiply %4163, %arg118 : tensor<576xbf16>
    %4165 = stablehlo.add %4162, %4164 : tensor<576xbf16>
    %cst_690 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4166 = stablehlo.broadcast_in_dim %cst_690, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4167 = stablehlo.multiply %4166, %3840 : tensor<576x9x128xbf16>
    %cst_691 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4168 = stablehlo.broadcast_in_dim %cst_691, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4169 = stablehlo.multiply %4168, %arg119 : tensor<576x9x128xbf16>
    %4170 = stablehlo.add %4167, %4169 : tensor<576x9x128xbf16>
    %cst_692 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4171 = stablehlo.broadcast_in_dim %cst_692, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4172 = stablehlo.multiply %4171, %3845 : tensor<9x128x576xbf16>
    %cst_693 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4173 = stablehlo.broadcast_in_dim %cst_693, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4174 = stablehlo.multiply %4173, %arg120 : tensor<9x128x576xbf16>
    %4175 = stablehlo.add %4172, %4174 : tensor<9x128x576xbf16>
    %cst_694 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4176 = stablehlo.broadcast_in_dim %cst_694, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4177 = stablehlo.multiply %4176, %3850 : tensor<576x9x128xbf16>
    %cst_695 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4178 = stablehlo.broadcast_in_dim %cst_695, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4179 = stablehlo.multiply %4178, %arg121 : tensor<576x9x128xbf16>
    %4180 = stablehlo.add %4177, %4179 : tensor<576x9x128xbf16>
    %cst_696 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4181 = stablehlo.broadcast_in_dim %cst_696, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4182 = stablehlo.multiply %4181, %3855 : tensor<576x9x128xbf16>
    %cst_697 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4183 = stablehlo.broadcast_in_dim %cst_697, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4184 = stablehlo.multiply %4183, %arg122 : tensor<576x9x128xbf16>
    %4185 = stablehlo.add %4182, %4184 : tensor<576x9x128xbf16>
    %cst_698 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4186 = stablehlo.broadcast_in_dim %cst_698, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4187 = stablehlo.multiply %4186, %3860 : tensor<576x2016xbf16>
    %cst_699 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4188 = stablehlo.broadcast_in_dim %cst_699, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4189 = stablehlo.multiply %4188, %arg123 : tensor<576x2016xbf16>
    %4190 = stablehlo.add %4187, %4189 : tensor<576x2016xbf16>
    %cst_700 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4191 = stablehlo.broadcast_in_dim %cst_700, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4192 = stablehlo.multiply %4191, %3865 : tensor<576x2016xbf16>
    %cst_701 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4193 = stablehlo.broadcast_in_dim %cst_701, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4194 = stablehlo.multiply %4193, %arg124 : tensor<576x2016xbf16>
    %4195 = stablehlo.add %4192, %4194 : tensor<576x2016xbf16>
    %cst_702 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4196 = stablehlo.broadcast_in_dim %cst_702, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4197 = stablehlo.multiply %4196, %3870 : tensor<2016x576xbf16>
    %cst_703 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4198 = stablehlo.broadcast_in_dim %cst_703, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4199 = stablehlo.multiply %4198, %arg125 : tensor<2016x576xbf16>
    %4200 = stablehlo.add %4197, %4199 : tensor<2016x576xbf16>
    %cst_704 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4201 = stablehlo.broadcast_in_dim %cst_704, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4202 = stablehlo.multiply %4201, %3875 : tensor<576xbf16>
    %cst_705 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4203 = stablehlo.broadcast_in_dim %cst_705, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4204 = stablehlo.multiply %4203, %arg126 : tensor<576xbf16>
    %4205 = stablehlo.add %4202, %4204 : tensor<576xbf16>
    %cst_706 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4206 = stablehlo.broadcast_in_dim %cst_706, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4207 = stablehlo.multiply %4206, %3880 : tensor<576xbf16>
    %cst_707 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4208 = stablehlo.broadcast_in_dim %cst_707, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4209 = stablehlo.multiply %4208, %arg127 : tensor<576xbf16>
    %4210 = stablehlo.add %4207, %4209 : tensor<576xbf16>
    %cst_708 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4211 = stablehlo.broadcast_in_dim %cst_708, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4212 = stablehlo.multiply %4211, %3885 : tensor<576x9x128xbf16>
    %cst_709 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4213 = stablehlo.broadcast_in_dim %cst_709, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4214 = stablehlo.multiply %4213, %arg128 : tensor<576x9x128xbf16>
    %4215 = stablehlo.add %4212, %4214 : tensor<576x9x128xbf16>
    %cst_710 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4216 = stablehlo.broadcast_in_dim %cst_710, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4217 = stablehlo.multiply %4216, %3890 : tensor<9x128x576xbf16>
    %cst_711 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4218 = stablehlo.broadcast_in_dim %cst_711, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4219 = stablehlo.multiply %4218, %arg129 : tensor<9x128x576xbf16>
    %4220 = stablehlo.add %4217, %4219 : tensor<9x128x576xbf16>
    %cst_712 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4221 = stablehlo.broadcast_in_dim %cst_712, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4222 = stablehlo.multiply %4221, %3895 : tensor<576x9x128xbf16>
    %cst_713 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4223 = stablehlo.broadcast_in_dim %cst_713, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4224 = stablehlo.multiply %4223, %arg130 : tensor<576x9x128xbf16>
    %4225 = stablehlo.add %4222, %4224 : tensor<576x9x128xbf16>
    %cst_714 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4226 = stablehlo.broadcast_in_dim %cst_714, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4227 = stablehlo.multiply %4226, %3900 : tensor<576x9x128xbf16>
    %cst_715 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4228 = stablehlo.broadcast_in_dim %cst_715, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4229 = stablehlo.multiply %4228, %arg131 : tensor<576x9x128xbf16>
    %4230 = stablehlo.add %4227, %4229 : tensor<576x9x128xbf16>
    %cst_716 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4231 = stablehlo.broadcast_in_dim %cst_716, dims = [] : (tensor<bf16>) -> tensor<576x128256xbf16>
    %4232 = stablehlo.multiply %4231, %3905 : tensor<576x128256xbf16>
    %cst_717 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4233 = stablehlo.broadcast_in_dim %cst_717, dims = [] : (tensor<bf16>) -> tensor<576x128256xbf16>
    %4234 = stablehlo.multiply %4233, %arg132 : tensor<576x128256xbf16>
    %4235 = stablehlo.add %4232, %4234 : tensor<576x128256xbf16>
    %cst_718 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %4236 = stablehlo.broadcast_in_dim %cst_718, dims = [] : (tensor<bf16>) -> tensor<128256x576xbf16>
    %4237 = stablehlo.multiply %4236, %3910 : tensor<128256x576xbf16>
    %cst_719 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %4238 = stablehlo.broadcast_in_dim %cst_719, dims = [] : (tensor<bf16>) -> tensor<128256x576xbf16>
    %4239 = stablehlo.multiply %4238, %arg133 : tensor<128256x576xbf16>
    %4240 = stablehlo.add %4237, %4239 : tensor<128256x576xbf16>
    %4241 = stablehlo.multiply %3585, %3585 : tensor<576xbf16>
    %cst_720 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4242 = stablehlo.broadcast_in_dim %cst_720, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4243 = stablehlo.multiply %4242, %4241 : tensor<576xbf16>
    %cst_721 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4244 = stablehlo.broadcast_in_dim %cst_721, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4245 = stablehlo.multiply %4244, %arg134 : tensor<576xbf16>
    %4246 = stablehlo.add %4243, %4245 : tensor<576xbf16>
    %4247 = stablehlo.multiply %3590, %3590 : tensor<576x2016xbf16>
    %cst_722 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4248 = stablehlo.broadcast_in_dim %cst_722, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4249 = stablehlo.multiply %4248, %4247 : tensor<576x2016xbf16>
    %cst_723 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4250 = stablehlo.broadcast_in_dim %cst_723, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4251 = stablehlo.multiply %4250, %arg135 : tensor<576x2016xbf16>
    %4252 = stablehlo.add %4249, %4251 : tensor<576x2016xbf16>
    %4253 = stablehlo.multiply %3595, %3595 : tensor<576x2016xbf16>
    %cst_724 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4254 = stablehlo.broadcast_in_dim %cst_724, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4255 = stablehlo.multiply %4254, %4253 : tensor<576x2016xbf16>
    %cst_725 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4256 = stablehlo.broadcast_in_dim %cst_725, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4257 = stablehlo.multiply %4256, %arg136 : tensor<576x2016xbf16>
    %4258 = stablehlo.add %4255, %4257 : tensor<576x2016xbf16>
    %4259 = stablehlo.multiply %3600, %3600 : tensor<2016x576xbf16>
    %cst_726 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4260 = stablehlo.broadcast_in_dim %cst_726, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4261 = stablehlo.multiply %4260, %4259 : tensor<2016x576xbf16>
    %cst_727 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4262 = stablehlo.broadcast_in_dim %cst_727, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4263 = stablehlo.multiply %4262, %arg137 : tensor<2016x576xbf16>
    %4264 = stablehlo.add %4261, %4263 : tensor<2016x576xbf16>
    %4265 = stablehlo.multiply %3605, %3605 : tensor<576xbf16>
    %cst_728 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4266 = stablehlo.broadcast_in_dim %cst_728, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4267 = stablehlo.multiply %4266, %4265 : tensor<576xbf16>
    %cst_729 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4268 = stablehlo.broadcast_in_dim %cst_729, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4269 = stablehlo.multiply %4268, %arg138 : tensor<576xbf16>
    %4270 = stablehlo.add %4267, %4269 : tensor<576xbf16>
    %4271 = stablehlo.multiply %3610, %3610 : tensor<576xbf16>
    %cst_730 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4272 = stablehlo.broadcast_in_dim %cst_730, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4273 = stablehlo.multiply %4272, %4271 : tensor<576xbf16>
    %cst_731 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4274 = stablehlo.broadcast_in_dim %cst_731, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4275 = stablehlo.multiply %4274, %arg139 : tensor<576xbf16>
    %4276 = stablehlo.add %4273, %4275 : tensor<576xbf16>
    %4277 = stablehlo.multiply %3615, %3615 : tensor<576x9x128xbf16>
    %cst_732 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4278 = stablehlo.broadcast_in_dim %cst_732, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4279 = stablehlo.multiply %4278, %4277 : tensor<576x9x128xbf16>
    %cst_733 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4280 = stablehlo.broadcast_in_dim %cst_733, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4281 = stablehlo.multiply %4280, %arg140 : tensor<576x9x128xbf16>
    %4282 = stablehlo.add %4279, %4281 : tensor<576x9x128xbf16>
    %4283 = stablehlo.multiply %3620, %3620 : tensor<9x128x576xbf16>
    %cst_734 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4284 = stablehlo.broadcast_in_dim %cst_734, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4285 = stablehlo.multiply %4284, %4283 : tensor<9x128x576xbf16>
    %cst_735 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4286 = stablehlo.broadcast_in_dim %cst_735, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4287 = stablehlo.multiply %4286, %arg141 : tensor<9x128x576xbf16>
    %4288 = stablehlo.add %4285, %4287 : tensor<9x128x576xbf16>
    %4289 = stablehlo.multiply %3625, %3625 : tensor<576x9x128xbf16>
    %cst_736 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4290 = stablehlo.broadcast_in_dim %cst_736, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4291 = stablehlo.multiply %4290, %4289 : tensor<576x9x128xbf16>
    %cst_737 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4292 = stablehlo.broadcast_in_dim %cst_737, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4293 = stablehlo.multiply %4292, %arg142 : tensor<576x9x128xbf16>
    %4294 = stablehlo.add %4291, %4293 : tensor<576x9x128xbf16>
    %4295 = stablehlo.multiply %3630, %3630 : tensor<576x9x128xbf16>
    %cst_738 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4296 = stablehlo.broadcast_in_dim %cst_738, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4297 = stablehlo.multiply %4296, %4295 : tensor<576x9x128xbf16>
    %cst_739 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4298 = stablehlo.broadcast_in_dim %cst_739, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4299 = stablehlo.multiply %4298, %arg143 : tensor<576x9x128xbf16>
    %4300 = stablehlo.add %4297, %4299 : tensor<576x9x128xbf16>
    %4301 = stablehlo.multiply %3635, %3635 : tensor<576x2016xbf16>
    %cst_740 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4302 = stablehlo.broadcast_in_dim %cst_740, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4303 = stablehlo.multiply %4302, %4301 : tensor<576x2016xbf16>
    %cst_741 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4304 = stablehlo.broadcast_in_dim %cst_741, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4305 = stablehlo.multiply %4304, %arg144 : tensor<576x2016xbf16>
    %4306 = stablehlo.add %4303, %4305 : tensor<576x2016xbf16>
    %4307 = stablehlo.multiply %3640, %3640 : tensor<576x2016xbf16>
    %cst_742 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4308 = stablehlo.broadcast_in_dim %cst_742, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4309 = stablehlo.multiply %4308, %4307 : tensor<576x2016xbf16>
    %cst_743 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4310 = stablehlo.broadcast_in_dim %cst_743, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4311 = stablehlo.multiply %4310, %arg145 : tensor<576x2016xbf16>
    %4312 = stablehlo.add %4309, %4311 : tensor<576x2016xbf16>
    %4313 = stablehlo.multiply %3645, %3645 : tensor<2016x576xbf16>
    %cst_744 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4314 = stablehlo.broadcast_in_dim %cst_744, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4315 = stablehlo.multiply %4314, %4313 : tensor<2016x576xbf16>
    %cst_745 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4316 = stablehlo.broadcast_in_dim %cst_745, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4317 = stablehlo.multiply %4316, %arg146 : tensor<2016x576xbf16>
    %4318 = stablehlo.add %4315, %4317 : tensor<2016x576xbf16>
    %4319 = stablehlo.multiply %3650, %3650 : tensor<576xbf16>
    %cst_746 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4320 = stablehlo.broadcast_in_dim %cst_746, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4321 = stablehlo.multiply %4320, %4319 : tensor<576xbf16>
    %cst_747 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4322 = stablehlo.broadcast_in_dim %cst_747, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4323 = stablehlo.multiply %4322, %arg147 : tensor<576xbf16>
    %4324 = stablehlo.add %4321, %4323 : tensor<576xbf16>
    %4325 = stablehlo.multiply %3655, %3655 : tensor<576xbf16>
    %cst_748 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4326 = stablehlo.broadcast_in_dim %cst_748, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4327 = stablehlo.multiply %4326, %4325 : tensor<576xbf16>
    %cst_749 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4328 = stablehlo.broadcast_in_dim %cst_749, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4329 = stablehlo.multiply %4328, %arg148 : tensor<576xbf16>
    %4330 = stablehlo.add %4327, %4329 : tensor<576xbf16>
    %4331 = stablehlo.multiply %3660, %3660 : tensor<576x9x128xbf16>
    %cst_750 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4332 = stablehlo.broadcast_in_dim %cst_750, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4333 = stablehlo.multiply %4332, %4331 : tensor<576x9x128xbf16>
    %cst_751 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4334 = stablehlo.broadcast_in_dim %cst_751, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4335 = stablehlo.multiply %4334, %arg149 : tensor<576x9x128xbf16>
    %4336 = stablehlo.add %4333, %4335 : tensor<576x9x128xbf16>
    %4337 = stablehlo.multiply %3665, %3665 : tensor<9x128x576xbf16>
    %cst_752 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4338 = stablehlo.broadcast_in_dim %cst_752, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4339 = stablehlo.multiply %4338, %4337 : tensor<9x128x576xbf16>
    %cst_753 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4340 = stablehlo.broadcast_in_dim %cst_753, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4341 = stablehlo.multiply %4340, %arg150 : tensor<9x128x576xbf16>
    %4342 = stablehlo.add %4339, %4341 : tensor<9x128x576xbf16>
    %4343 = stablehlo.multiply %3670, %3670 : tensor<576x9x128xbf16>
    %cst_754 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4344 = stablehlo.broadcast_in_dim %cst_754, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4345 = stablehlo.multiply %4344, %4343 : tensor<576x9x128xbf16>
    %cst_755 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4346 = stablehlo.broadcast_in_dim %cst_755, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4347 = stablehlo.multiply %4346, %arg151 : tensor<576x9x128xbf16>
    %4348 = stablehlo.add %4345, %4347 : tensor<576x9x128xbf16>
    %4349 = stablehlo.multiply %3675, %3675 : tensor<576x9x128xbf16>
    %cst_756 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4350 = stablehlo.broadcast_in_dim %cst_756, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4351 = stablehlo.multiply %4350, %4349 : tensor<576x9x128xbf16>
    %cst_757 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4352 = stablehlo.broadcast_in_dim %cst_757, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4353 = stablehlo.multiply %4352, %arg152 : tensor<576x9x128xbf16>
    %4354 = stablehlo.add %4351, %4353 : tensor<576x9x128xbf16>
    %4355 = stablehlo.multiply %3680, %3680 : tensor<576x2016xbf16>
    %cst_758 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4356 = stablehlo.broadcast_in_dim %cst_758, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4357 = stablehlo.multiply %4356, %4355 : tensor<576x2016xbf16>
    %cst_759 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4358 = stablehlo.broadcast_in_dim %cst_759, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4359 = stablehlo.multiply %4358, %arg153 : tensor<576x2016xbf16>
    %4360 = stablehlo.add %4357, %4359 : tensor<576x2016xbf16>
    %4361 = stablehlo.multiply %3685, %3685 : tensor<576x2016xbf16>
    %cst_760 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4362 = stablehlo.broadcast_in_dim %cst_760, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4363 = stablehlo.multiply %4362, %4361 : tensor<576x2016xbf16>
    %cst_761 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4364 = stablehlo.broadcast_in_dim %cst_761, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4365 = stablehlo.multiply %4364, %arg154 : tensor<576x2016xbf16>
    %4366 = stablehlo.add %4363, %4365 : tensor<576x2016xbf16>
    %4367 = stablehlo.multiply %3690, %3690 : tensor<2016x576xbf16>
    %cst_762 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4368 = stablehlo.broadcast_in_dim %cst_762, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4369 = stablehlo.multiply %4368, %4367 : tensor<2016x576xbf16>
    %cst_763 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4370 = stablehlo.broadcast_in_dim %cst_763, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4371 = stablehlo.multiply %4370, %arg155 : tensor<2016x576xbf16>
    %4372 = stablehlo.add %4369, %4371 : tensor<2016x576xbf16>
    %4373 = stablehlo.multiply %3695, %3695 : tensor<576xbf16>
    %cst_764 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4374 = stablehlo.broadcast_in_dim %cst_764, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4375 = stablehlo.multiply %4374, %4373 : tensor<576xbf16>
    %cst_765 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4376 = stablehlo.broadcast_in_dim %cst_765, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4377 = stablehlo.multiply %4376, %arg156 : tensor<576xbf16>
    %4378 = stablehlo.add %4375, %4377 : tensor<576xbf16>
    %4379 = stablehlo.multiply %3700, %3700 : tensor<576xbf16>
    %cst_766 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4380 = stablehlo.broadcast_in_dim %cst_766, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4381 = stablehlo.multiply %4380, %4379 : tensor<576xbf16>
    %cst_767 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4382 = stablehlo.broadcast_in_dim %cst_767, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4383 = stablehlo.multiply %4382, %arg157 : tensor<576xbf16>
    %4384 = stablehlo.add %4381, %4383 : tensor<576xbf16>
    %4385 = stablehlo.multiply %3705, %3705 : tensor<576x9x128xbf16>
    %cst_768 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4386 = stablehlo.broadcast_in_dim %cst_768, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4387 = stablehlo.multiply %4386, %4385 : tensor<576x9x128xbf16>
    %cst_769 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4388 = stablehlo.broadcast_in_dim %cst_769, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4389 = stablehlo.multiply %4388, %arg158 : tensor<576x9x128xbf16>
    %4390 = stablehlo.add %4387, %4389 : tensor<576x9x128xbf16>
    %4391 = stablehlo.multiply %3710, %3710 : tensor<9x128x576xbf16>
    %cst_770 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4392 = stablehlo.broadcast_in_dim %cst_770, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4393 = stablehlo.multiply %4392, %4391 : tensor<9x128x576xbf16>
    %cst_771 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4394 = stablehlo.broadcast_in_dim %cst_771, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4395 = stablehlo.multiply %4394, %arg159 : tensor<9x128x576xbf16>
    %4396 = stablehlo.add %4393, %4395 : tensor<9x128x576xbf16>
    %4397 = stablehlo.multiply %3715, %3715 : tensor<576x9x128xbf16>
    %cst_772 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4398 = stablehlo.broadcast_in_dim %cst_772, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4399 = stablehlo.multiply %4398, %4397 : tensor<576x9x128xbf16>
    %cst_773 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4400 = stablehlo.broadcast_in_dim %cst_773, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4401 = stablehlo.multiply %4400, %arg160 : tensor<576x9x128xbf16>
    %4402 = stablehlo.add %4399, %4401 : tensor<576x9x128xbf16>
    %4403 = stablehlo.multiply %3720, %3720 : tensor<576x9x128xbf16>
    %cst_774 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4404 = stablehlo.broadcast_in_dim %cst_774, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4405 = stablehlo.multiply %4404, %4403 : tensor<576x9x128xbf16>
    %cst_775 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4406 = stablehlo.broadcast_in_dim %cst_775, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4407 = stablehlo.multiply %4406, %arg161 : tensor<576x9x128xbf16>
    %4408 = stablehlo.add %4405, %4407 : tensor<576x9x128xbf16>
    %4409 = stablehlo.multiply %3725, %3725 : tensor<576x2016xbf16>
    %cst_776 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4410 = stablehlo.broadcast_in_dim %cst_776, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4411 = stablehlo.multiply %4410, %4409 : tensor<576x2016xbf16>
    %cst_777 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4412 = stablehlo.broadcast_in_dim %cst_777, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4413 = stablehlo.multiply %4412, %arg162 : tensor<576x2016xbf16>
    %4414 = stablehlo.add %4411, %4413 : tensor<576x2016xbf16>
    %4415 = stablehlo.multiply %3730, %3730 : tensor<576x2016xbf16>
    %cst_778 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4416 = stablehlo.broadcast_in_dim %cst_778, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4417 = stablehlo.multiply %4416, %4415 : tensor<576x2016xbf16>
    %cst_779 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4418 = stablehlo.broadcast_in_dim %cst_779, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4419 = stablehlo.multiply %4418, %arg163 : tensor<576x2016xbf16>
    %4420 = stablehlo.add %4417, %4419 : tensor<576x2016xbf16>
    %4421 = stablehlo.multiply %3735, %3735 : tensor<2016x576xbf16>
    %cst_780 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4422 = stablehlo.broadcast_in_dim %cst_780, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4423 = stablehlo.multiply %4422, %4421 : tensor<2016x576xbf16>
    %cst_781 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4424 = stablehlo.broadcast_in_dim %cst_781, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4425 = stablehlo.multiply %4424, %arg164 : tensor<2016x576xbf16>
    %4426 = stablehlo.add %4423, %4425 : tensor<2016x576xbf16>
    %4427 = stablehlo.multiply %3740, %3740 : tensor<576xbf16>
    %cst_782 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4428 = stablehlo.broadcast_in_dim %cst_782, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4429 = stablehlo.multiply %4428, %4427 : tensor<576xbf16>
    %cst_783 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4430 = stablehlo.broadcast_in_dim %cst_783, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4431 = stablehlo.multiply %4430, %arg165 : tensor<576xbf16>
    %4432 = stablehlo.add %4429, %4431 : tensor<576xbf16>
    %4433 = stablehlo.multiply %3745, %3745 : tensor<576xbf16>
    %cst_784 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4434 = stablehlo.broadcast_in_dim %cst_784, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4435 = stablehlo.multiply %4434, %4433 : tensor<576xbf16>
    %cst_785 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4436 = stablehlo.broadcast_in_dim %cst_785, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4437 = stablehlo.multiply %4436, %arg166 : tensor<576xbf16>
    %4438 = stablehlo.add %4435, %4437 : tensor<576xbf16>
    %4439 = stablehlo.multiply %3750, %3750 : tensor<576x9x128xbf16>
    %cst_786 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4440 = stablehlo.broadcast_in_dim %cst_786, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4441 = stablehlo.multiply %4440, %4439 : tensor<576x9x128xbf16>
    %cst_787 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4442 = stablehlo.broadcast_in_dim %cst_787, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4443 = stablehlo.multiply %4442, %arg167 : tensor<576x9x128xbf16>
    %4444 = stablehlo.add %4441, %4443 : tensor<576x9x128xbf16>
    %4445 = stablehlo.multiply %3755, %3755 : tensor<9x128x576xbf16>
    %cst_788 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4446 = stablehlo.broadcast_in_dim %cst_788, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4447 = stablehlo.multiply %4446, %4445 : tensor<9x128x576xbf16>
    %cst_789 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4448 = stablehlo.broadcast_in_dim %cst_789, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4449 = stablehlo.multiply %4448, %arg168 : tensor<9x128x576xbf16>
    %4450 = stablehlo.add %4447, %4449 : tensor<9x128x576xbf16>
    %4451 = stablehlo.multiply %3760, %3760 : tensor<576x9x128xbf16>
    %cst_790 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4452 = stablehlo.broadcast_in_dim %cst_790, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4453 = stablehlo.multiply %4452, %4451 : tensor<576x9x128xbf16>
    %cst_791 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4454 = stablehlo.broadcast_in_dim %cst_791, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4455 = stablehlo.multiply %4454, %arg169 : tensor<576x9x128xbf16>
    %4456 = stablehlo.add %4453, %4455 : tensor<576x9x128xbf16>
    %4457 = stablehlo.multiply %3765, %3765 : tensor<576x9x128xbf16>
    %cst_792 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4458 = stablehlo.broadcast_in_dim %cst_792, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4459 = stablehlo.multiply %4458, %4457 : tensor<576x9x128xbf16>
    %cst_793 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4460 = stablehlo.broadcast_in_dim %cst_793, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4461 = stablehlo.multiply %4460, %arg170 : tensor<576x9x128xbf16>
    %4462 = stablehlo.add %4459, %4461 : tensor<576x9x128xbf16>
    %4463 = stablehlo.multiply %3770, %3770 : tensor<576x2016xbf16>
    %cst_794 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4464 = stablehlo.broadcast_in_dim %cst_794, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4465 = stablehlo.multiply %4464, %4463 : tensor<576x2016xbf16>
    %cst_795 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4466 = stablehlo.broadcast_in_dim %cst_795, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4467 = stablehlo.multiply %4466, %arg171 : tensor<576x2016xbf16>
    %4468 = stablehlo.add %4465, %4467 : tensor<576x2016xbf16>
    %4469 = stablehlo.multiply %3775, %3775 : tensor<576x2016xbf16>
    %cst_796 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4470 = stablehlo.broadcast_in_dim %cst_796, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4471 = stablehlo.multiply %4470, %4469 : tensor<576x2016xbf16>
    %cst_797 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4472 = stablehlo.broadcast_in_dim %cst_797, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4473 = stablehlo.multiply %4472, %arg172 : tensor<576x2016xbf16>
    %4474 = stablehlo.add %4471, %4473 : tensor<576x2016xbf16>
    %4475 = stablehlo.multiply %3780, %3780 : tensor<2016x576xbf16>
    %cst_798 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4476 = stablehlo.broadcast_in_dim %cst_798, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4477 = stablehlo.multiply %4476, %4475 : tensor<2016x576xbf16>
    %cst_799 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4478 = stablehlo.broadcast_in_dim %cst_799, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4479 = stablehlo.multiply %4478, %arg173 : tensor<2016x576xbf16>
    %4480 = stablehlo.add %4477, %4479 : tensor<2016x576xbf16>
    %4481 = stablehlo.multiply %3785, %3785 : tensor<576xbf16>
    %cst_800 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4482 = stablehlo.broadcast_in_dim %cst_800, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4483 = stablehlo.multiply %4482, %4481 : tensor<576xbf16>
    %cst_801 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4484 = stablehlo.broadcast_in_dim %cst_801, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4485 = stablehlo.multiply %4484, %arg174 : tensor<576xbf16>
    %4486 = stablehlo.add %4483, %4485 : tensor<576xbf16>
    %4487 = stablehlo.multiply %3790, %3790 : tensor<576xbf16>
    %cst_802 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4488 = stablehlo.broadcast_in_dim %cst_802, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4489 = stablehlo.multiply %4488, %4487 : tensor<576xbf16>
    %cst_803 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4490 = stablehlo.broadcast_in_dim %cst_803, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4491 = stablehlo.multiply %4490, %arg175 : tensor<576xbf16>
    %4492 = stablehlo.add %4489, %4491 : tensor<576xbf16>
    %4493 = stablehlo.multiply %3795, %3795 : tensor<576x9x128xbf16>
    %cst_804 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4494 = stablehlo.broadcast_in_dim %cst_804, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4495 = stablehlo.multiply %4494, %4493 : tensor<576x9x128xbf16>
    %cst_805 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4496 = stablehlo.broadcast_in_dim %cst_805, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4497 = stablehlo.multiply %4496, %arg176 : tensor<576x9x128xbf16>
    %4498 = stablehlo.add %4495, %4497 : tensor<576x9x128xbf16>
    %4499 = stablehlo.multiply %3800, %3800 : tensor<9x128x576xbf16>
    %cst_806 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4500 = stablehlo.broadcast_in_dim %cst_806, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4501 = stablehlo.multiply %4500, %4499 : tensor<9x128x576xbf16>
    %cst_807 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4502 = stablehlo.broadcast_in_dim %cst_807, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4503 = stablehlo.multiply %4502, %arg177 : tensor<9x128x576xbf16>
    %4504 = stablehlo.add %4501, %4503 : tensor<9x128x576xbf16>
    %4505 = stablehlo.multiply %3805, %3805 : tensor<576x9x128xbf16>
    %cst_808 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4506 = stablehlo.broadcast_in_dim %cst_808, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4507 = stablehlo.multiply %4506, %4505 : tensor<576x9x128xbf16>
    %cst_809 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4508 = stablehlo.broadcast_in_dim %cst_809, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4509 = stablehlo.multiply %4508, %arg178 : tensor<576x9x128xbf16>
    %4510 = stablehlo.add %4507, %4509 : tensor<576x9x128xbf16>
    %4511 = stablehlo.multiply %3810, %3810 : tensor<576x9x128xbf16>
    %cst_810 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4512 = stablehlo.broadcast_in_dim %cst_810, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4513 = stablehlo.multiply %4512, %4511 : tensor<576x9x128xbf16>
    %cst_811 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4514 = stablehlo.broadcast_in_dim %cst_811, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4515 = stablehlo.multiply %4514, %arg179 : tensor<576x9x128xbf16>
    %4516 = stablehlo.add %4513, %4515 : tensor<576x9x128xbf16>
    %4517 = stablehlo.multiply %3815, %3815 : tensor<576x2016xbf16>
    %cst_812 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4518 = stablehlo.broadcast_in_dim %cst_812, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4519 = stablehlo.multiply %4518, %4517 : tensor<576x2016xbf16>
    %cst_813 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4520 = stablehlo.broadcast_in_dim %cst_813, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4521 = stablehlo.multiply %4520, %arg180 : tensor<576x2016xbf16>
    %4522 = stablehlo.add %4519, %4521 : tensor<576x2016xbf16>
    %4523 = stablehlo.multiply %3820, %3820 : tensor<576x2016xbf16>
    %cst_814 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4524 = stablehlo.broadcast_in_dim %cst_814, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4525 = stablehlo.multiply %4524, %4523 : tensor<576x2016xbf16>
    %cst_815 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4526 = stablehlo.broadcast_in_dim %cst_815, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4527 = stablehlo.multiply %4526, %arg181 : tensor<576x2016xbf16>
    %4528 = stablehlo.add %4525, %4527 : tensor<576x2016xbf16>
    %4529 = stablehlo.multiply %3825, %3825 : tensor<2016x576xbf16>
    %cst_816 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4530 = stablehlo.broadcast_in_dim %cst_816, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4531 = stablehlo.multiply %4530, %4529 : tensor<2016x576xbf16>
    %cst_817 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4532 = stablehlo.broadcast_in_dim %cst_817, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4533 = stablehlo.multiply %4532, %arg182 : tensor<2016x576xbf16>
    %4534 = stablehlo.add %4531, %4533 : tensor<2016x576xbf16>
    %4535 = stablehlo.multiply %3830, %3830 : tensor<576xbf16>
    %cst_818 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4536 = stablehlo.broadcast_in_dim %cst_818, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4537 = stablehlo.multiply %4536, %4535 : tensor<576xbf16>
    %cst_819 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4538 = stablehlo.broadcast_in_dim %cst_819, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4539 = stablehlo.multiply %4538, %arg183 : tensor<576xbf16>
    %4540 = stablehlo.add %4537, %4539 : tensor<576xbf16>
    %4541 = stablehlo.multiply %3835, %3835 : tensor<576xbf16>
    %cst_820 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4542 = stablehlo.broadcast_in_dim %cst_820, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4543 = stablehlo.multiply %4542, %4541 : tensor<576xbf16>
    %cst_821 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4544 = stablehlo.broadcast_in_dim %cst_821, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4545 = stablehlo.multiply %4544, %arg184 : tensor<576xbf16>
    %4546 = stablehlo.add %4543, %4545 : tensor<576xbf16>
    %4547 = stablehlo.multiply %3840, %3840 : tensor<576x9x128xbf16>
    %cst_822 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4548 = stablehlo.broadcast_in_dim %cst_822, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4549 = stablehlo.multiply %4548, %4547 : tensor<576x9x128xbf16>
    %cst_823 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4550 = stablehlo.broadcast_in_dim %cst_823, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4551 = stablehlo.multiply %4550, %arg185 : tensor<576x9x128xbf16>
    %4552 = stablehlo.add %4549, %4551 : tensor<576x9x128xbf16>
    %4553 = stablehlo.multiply %3845, %3845 : tensor<9x128x576xbf16>
    %cst_824 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4554 = stablehlo.broadcast_in_dim %cst_824, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4555 = stablehlo.multiply %4554, %4553 : tensor<9x128x576xbf16>
    %cst_825 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4556 = stablehlo.broadcast_in_dim %cst_825, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4557 = stablehlo.multiply %4556, %arg186 : tensor<9x128x576xbf16>
    %4558 = stablehlo.add %4555, %4557 : tensor<9x128x576xbf16>
    %4559 = stablehlo.multiply %3850, %3850 : tensor<576x9x128xbf16>
    %cst_826 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4560 = stablehlo.broadcast_in_dim %cst_826, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4561 = stablehlo.multiply %4560, %4559 : tensor<576x9x128xbf16>
    %cst_827 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4562 = stablehlo.broadcast_in_dim %cst_827, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4563 = stablehlo.multiply %4562, %arg187 : tensor<576x9x128xbf16>
    %4564 = stablehlo.add %4561, %4563 : tensor<576x9x128xbf16>
    %4565 = stablehlo.multiply %3855, %3855 : tensor<576x9x128xbf16>
    %cst_828 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4566 = stablehlo.broadcast_in_dim %cst_828, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4567 = stablehlo.multiply %4566, %4565 : tensor<576x9x128xbf16>
    %cst_829 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4568 = stablehlo.broadcast_in_dim %cst_829, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4569 = stablehlo.multiply %4568, %arg188 : tensor<576x9x128xbf16>
    %4570 = stablehlo.add %4567, %4569 : tensor<576x9x128xbf16>
    %4571 = stablehlo.multiply %3860, %3860 : tensor<576x2016xbf16>
    %cst_830 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4572 = stablehlo.broadcast_in_dim %cst_830, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4573 = stablehlo.multiply %4572, %4571 : tensor<576x2016xbf16>
    %cst_831 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4574 = stablehlo.broadcast_in_dim %cst_831, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4575 = stablehlo.multiply %4574, %arg189 : tensor<576x2016xbf16>
    %4576 = stablehlo.add %4573, %4575 : tensor<576x2016xbf16>
    %4577 = stablehlo.multiply %3865, %3865 : tensor<576x2016xbf16>
    %cst_832 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4578 = stablehlo.broadcast_in_dim %cst_832, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4579 = stablehlo.multiply %4578, %4577 : tensor<576x2016xbf16>
    %cst_833 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4580 = stablehlo.broadcast_in_dim %cst_833, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4581 = stablehlo.multiply %4580, %arg190 : tensor<576x2016xbf16>
    %4582 = stablehlo.add %4579, %4581 : tensor<576x2016xbf16>
    %4583 = stablehlo.multiply %3870, %3870 : tensor<2016x576xbf16>
    %cst_834 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4584 = stablehlo.broadcast_in_dim %cst_834, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4585 = stablehlo.multiply %4584, %4583 : tensor<2016x576xbf16>
    %cst_835 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4586 = stablehlo.broadcast_in_dim %cst_835, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4587 = stablehlo.multiply %4586, %arg191 : tensor<2016x576xbf16>
    %4588 = stablehlo.add %4585, %4587 : tensor<2016x576xbf16>
    %4589 = stablehlo.multiply %3875, %3875 : tensor<576xbf16>
    %cst_836 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4590 = stablehlo.broadcast_in_dim %cst_836, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4591 = stablehlo.multiply %4590, %4589 : tensor<576xbf16>
    %cst_837 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4592 = stablehlo.broadcast_in_dim %cst_837, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4593 = stablehlo.multiply %4592, %arg192 : tensor<576xbf16>
    %4594 = stablehlo.add %4591, %4593 : tensor<576xbf16>
    %4595 = stablehlo.multiply %3880, %3880 : tensor<576xbf16>
    %cst_838 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4596 = stablehlo.broadcast_in_dim %cst_838, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4597 = stablehlo.multiply %4596, %4595 : tensor<576xbf16>
    %cst_839 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4598 = stablehlo.broadcast_in_dim %cst_839, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4599 = stablehlo.multiply %4598, %arg193 : tensor<576xbf16>
    %4600 = stablehlo.add %4597, %4599 : tensor<576xbf16>
    %4601 = stablehlo.multiply %3885, %3885 : tensor<576x9x128xbf16>
    %cst_840 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4602 = stablehlo.broadcast_in_dim %cst_840, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4603 = stablehlo.multiply %4602, %4601 : tensor<576x9x128xbf16>
    %cst_841 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4604 = stablehlo.broadcast_in_dim %cst_841, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4605 = stablehlo.multiply %4604, %arg194 : tensor<576x9x128xbf16>
    %4606 = stablehlo.add %4603, %4605 : tensor<576x9x128xbf16>
    %4607 = stablehlo.multiply %3890, %3890 : tensor<9x128x576xbf16>
    %cst_842 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4608 = stablehlo.broadcast_in_dim %cst_842, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4609 = stablehlo.multiply %4608, %4607 : tensor<9x128x576xbf16>
    %cst_843 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4610 = stablehlo.broadcast_in_dim %cst_843, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4611 = stablehlo.multiply %4610, %arg195 : tensor<9x128x576xbf16>
    %4612 = stablehlo.add %4609, %4611 : tensor<9x128x576xbf16>
    %4613 = stablehlo.multiply %3895, %3895 : tensor<576x9x128xbf16>
    %cst_844 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4614 = stablehlo.broadcast_in_dim %cst_844, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4615 = stablehlo.multiply %4614, %4613 : tensor<576x9x128xbf16>
    %cst_845 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4616 = stablehlo.broadcast_in_dim %cst_845, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4617 = stablehlo.multiply %4616, %arg196 : tensor<576x9x128xbf16>
    %4618 = stablehlo.add %4615, %4617 : tensor<576x9x128xbf16>
    %4619 = stablehlo.multiply %3900, %3900 : tensor<576x9x128xbf16>
    %cst_846 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4620 = stablehlo.broadcast_in_dim %cst_846, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4621 = stablehlo.multiply %4620, %4619 : tensor<576x9x128xbf16>
    %cst_847 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4622 = stablehlo.broadcast_in_dim %cst_847, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4623 = stablehlo.multiply %4622, %arg197 : tensor<576x9x128xbf16>
    %4624 = stablehlo.add %4621, %4623 : tensor<576x9x128xbf16>
    %4625 = stablehlo.multiply %3905, %3905 : tensor<576x128256xbf16>
    %cst_848 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4626 = stablehlo.broadcast_in_dim %cst_848, dims = [] : (tensor<bf16>) -> tensor<576x128256xbf16>
    %4627 = stablehlo.multiply %4626, %4625 : tensor<576x128256xbf16>
    %cst_849 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4628 = stablehlo.broadcast_in_dim %cst_849, dims = [] : (tensor<bf16>) -> tensor<576x128256xbf16>
    %4629 = stablehlo.multiply %4628, %arg198 : tensor<576x128256xbf16>
    %4630 = stablehlo.add %4627, %4629 : tensor<576x128256xbf16>
    %4631 = stablehlo.multiply %3910, %3910 : tensor<128256x576xbf16>
    %cst_850 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %4632 = stablehlo.broadcast_in_dim %cst_850, dims = [] : (tensor<bf16>) -> tensor<128256x576xbf16>
    %4633 = stablehlo.multiply %4632, %4631 : tensor<128256x576xbf16>
    %cst_851 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %4634 = stablehlo.broadcast_in_dim %cst_851, dims = [] : (tensor<bf16>) -> tensor<128256x576xbf16>
    %4635 = stablehlo.multiply %4634, %arg199 : tensor<128256x576xbf16>
    %4636 = stablehlo.add %4633, %4635 : tensor<128256x576xbf16>
    %c_852 = stablehlo.constant dense<2147483647> : tensor<i32>
    %4637 = stablehlo.compare  LT, %arg67, %c_852,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_853 = stablehlo.constant dense<1> : tensor<i32>
    %4638 = stablehlo.add %arg67, %c_853 : tensor<i32>
    %c_854 = stablehlo.constant dense<2147483647> : tensor<i32>
    %4639 = call @_where_260(%4637, %4638, %c_854) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %cst_855 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %4640 = stablehlo.convert %4639 : (tensor<i32>) -> tensor<f32>
    %4641 = stablehlo.power %cst_855, %4640 : tensor<f32>
    %cst_856 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4642 = stablehlo.subtract %cst_856, %4641 : tensor<f32>
    %4643 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4644 = stablehlo.broadcast_in_dim %4643, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4645 = stablehlo.divide %3915, %4644 : tensor<576xbf16>
    %4646 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4647 = stablehlo.broadcast_in_dim %4646, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4648 = stablehlo.divide %3920, %4647 : tensor<576x2016xbf16>
    %4649 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4650 = stablehlo.broadcast_in_dim %4649, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4651 = stablehlo.divide %3925, %4650 : tensor<576x2016xbf16>
    %4652 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4653 = stablehlo.broadcast_in_dim %4652, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4654 = stablehlo.divide %3930, %4653 : tensor<2016x576xbf16>
    %4655 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4656 = stablehlo.broadcast_in_dim %4655, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4657 = stablehlo.divide %3935, %4656 : tensor<576xbf16>
    %4658 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4659 = stablehlo.broadcast_in_dim %4658, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4660 = stablehlo.divide %3940, %4659 : tensor<576xbf16>
    %4661 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4662 = stablehlo.broadcast_in_dim %4661, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4663 = stablehlo.divide %3945, %4662 : tensor<576x9x128xbf16>
    %4664 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4665 = stablehlo.broadcast_in_dim %4664, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4666 = stablehlo.divide %3950, %4665 : tensor<9x128x576xbf16>
    %4667 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4668 = stablehlo.broadcast_in_dim %4667, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4669 = stablehlo.divide %3955, %4668 : tensor<576x9x128xbf16>
    %4670 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4671 = stablehlo.broadcast_in_dim %4670, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4672 = stablehlo.divide %3960, %4671 : tensor<576x9x128xbf16>
    %4673 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4674 = stablehlo.broadcast_in_dim %4673, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4675 = stablehlo.divide %3965, %4674 : tensor<576x2016xbf16>
    %4676 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4677 = stablehlo.broadcast_in_dim %4676, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4678 = stablehlo.divide %3970, %4677 : tensor<576x2016xbf16>
    %4679 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4680 = stablehlo.broadcast_in_dim %4679, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4681 = stablehlo.divide %3975, %4680 : tensor<2016x576xbf16>
    %4682 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4683 = stablehlo.broadcast_in_dim %4682, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4684 = stablehlo.divide %3980, %4683 : tensor<576xbf16>
    %4685 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4686 = stablehlo.broadcast_in_dim %4685, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4687 = stablehlo.divide %3985, %4686 : tensor<576xbf16>
    %4688 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4689 = stablehlo.broadcast_in_dim %4688, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4690 = stablehlo.divide %3990, %4689 : tensor<576x9x128xbf16>
    %4691 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4692 = stablehlo.broadcast_in_dim %4691, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4693 = stablehlo.divide %3995, %4692 : tensor<9x128x576xbf16>
    %4694 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4695 = stablehlo.broadcast_in_dim %4694, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4696 = stablehlo.divide %4000, %4695 : tensor<576x9x128xbf16>
    %4697 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4698 = stablehlo.broadcast_in_dim %4697, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4699 = stablehlo.divide %4005, %4698 : tensor<576x9x128xbf16>
    %4700 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4701 = stablehlo.broadcast_in_dim %4700, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4702 = stablehlo.divide %4010, %4701 : tensor<576x2016xbf16>
    %4703 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4704 = stablehlo.broadcast_in_dim %4703, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4705 = stablehlo.divide %4015, %4704 : tensor<576x2016xbf16>
    %4706 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4707 = stablehlo.broadcast_in_dim %4706, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4708 = stablehlo.divide %4020, %4707 : tensor<2016x576xbf16>
    %4709 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4710 = stablehlo.broadcast_in_dim %4709, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4711 = stablehlo.divide %4025, %4710 : tensor<576xbf16>
    %4712 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4713 = stablehlo.broadcast_in_dim %4712, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4714 = stablehlo.divide %4030, %4713 : tensor<576xbf16>
    %4715 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4716 = stablehlo.broadcast_in_dim %4715, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4717 = stablehlo.divide %4035, %4716 : tensor<576x9x128xbf16>
    %4718 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4719 = stablehlo.broadcast_in_dim %4718, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4720 = stablehlo.divide %4040, %4719 : tensor<9x128x576xbf16>
    %4721 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4722 = stablehlo.broadcast_in_dim %4721, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4723 = stablehlo.divide %4045, %4722 : tensor<576x9x128xbf16>
    %4724 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4725 = stablehlo.broadcast_in_dim %4724, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4726 = stablehlo.divide %4050, %4725 : tensor<576x9x128xbf16>
    %4727 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4728 = stablehlo.broadcast_in_dim %4727, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4729 = stablehlo.divide %4055, %4728 : tensor<576x2016xbf16>
    %4730 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4731 = stablehlo.broadcast_in_dim %4730, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4732 = stablehlo.divide %4060, %4731 : tensor<576x2016xbf16>
    %4733 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4734 = stablehlo.broadcast_in_dim %4733, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4735 = stablehlo.divide %4065, %4734 : tensor<2016x576xbf16>
    %4736 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4737 = stablehlo.broadcast_in_dim %4736, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4738 = stablehlo.divide %4070, %4737 : tensor<576xbf16>
    %4739 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4740 = stablehlo.broadcast_in_dim %4739, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4741 = stablehlo.divide %4075, %4740 : tensor<576xbf16>
    %4742 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4743 = stablehlo.broadcast_in_dim %4742, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4744 = stablehlo.divide %4080, %4743 : tensor<576x9x128xbf16>
    %4745 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4746 = stablehlo.broadcast_in_dim %4745, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4747 = stablehlo.divide %4085, %4746 : tensor<9x128x576xbf16>
    %4748 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4749 = stablehlo.broadcast_in_dim %4748, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4750 = stablehlo.divide %4090, %4749 : tensor<576x9x128xbf16>
    %4751 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4752 = stablehlo.broadcast_in_dim %4751, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4753 = stablehlo.divide %4095, %4752 : tensor<576x9x128xbf16>
    %4754 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4755 = stablehlo.broadcast_in_dim %4754, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4756 = stablehlo.divide %4100, %4755 : tensor<576x2016xbf16>
    %4757 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4758 = stablehlo.broadcast_in_dim %4757, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4759 = stablehlo.divide %4105, %4758 : tensor<576x2016xbf16>
    %4760 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4761 = stablehlo.broadcast_in_dim %4760, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4762 = stablehlo.divide %4110, %4761 : tensor<2016x576xbf16>
    %4763 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4764 = stablehlo.broadcast_in_dim %4763, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4765 = stablehlo.divide %4115, %4764 : tensor<576xbf16>
    %4766 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4767 = stablehlo.broadcast_in_dim %4766, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4768 = stablehlo.divide %4120, %4767 : tensor<576xbf16>
    %4769 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4770 = stablehlo.broadcast_in_dim %4769, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4771 = stablehlo.divide %4125, %4770 : tensor<576x9x128xbf16>
    %4772 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4773 = stablehlo.broadcast_in_dim %4772, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4774 = stablehlo.divide %4130, %4773 : tensor<9x128x576xbf16>
    %4775 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4776 = stablehlo.broadcast_in_dim %4775, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4777 = stablehlo.divide %4135, %4776 : tensor<576x9x128xbf16>
    %4778 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4779 = stablehlo.broadcast_in_dim %4778, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4780 = stablehlo.divide %4140, %4779 : tensor<576x9x128xbf16>
    %4781 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4782 = stablehlo.broadcast_in_dim %4781, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4783 = stablehlo.divide %4145, %4782 : tensor<576x2016xbf16>
    %4784 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4785 = stablehlo.broadcast_in_dim %4784, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4786 = stablehlo.divide %4150, %4785 : tensor<576x2016xbf16>
    %4787 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4788 = stablehlo.broadcast_in_dim %4787, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4789 = stablehlo.divide %4155, %4788 : tensor<2016x576xbf16>
    %4790 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4791 = stablehlo.broadcast_in_dim %4790, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4792 = stablehlo.divide %4160, %4791 : tensor<576xbf16>
    %4793 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4794 = stablehlo.broadcast_in_dim %4793, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4795 = stablehlo.divide %4165, %4794 : tensor<576xbf16>
    %4796 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4797 = stablehlo.broadcast_in_dim %4796, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4798 = stablehlo.divide %4170, %4797 : tensor<576x9x128xbf16>
    %4799 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4800 = stablehlo.broadcast_in_dim %4799, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4801 = stablehlo.divide %4175, %4800 : tensor<9x128x576xbf16>
    %4802 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4803 = stablehlo.broadcast_in_dim %4802, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4804 = stablehlo.divide %4180, %4803 : tensor<576x9x128xbf16>
    %4805 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4806 = stablehlo.broadcast_in_dim %4805, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4807 = stablehlo.divide %4185, %4806 : tensor<576x9x128xbf16>
    %4808 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4809 = stablehlo.broadcast_in_dim %4808, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4810 = stablehlo.divide %4190, %4809 : tensor<576x2016xbf16>
    %4811 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4812 = stablehlo.broadcast_in_dim %4811, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4813 = stablehlo.divide %4195, %4812 : tensor<576x2016xbf16>
    %4814 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4815 = stablehlo.broadcast_in_dim %4814, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4816 = stablehlo.divide %4200, %4815 : tensor<2016x576xbf16>
    %4817 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4818 = stablehlo.broadcast_in_dim %4817, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4819 = stablehlo.divide %4205, %4818 : tensor<576xbf16>
    %4820 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4821 = stablehlo.broadcast_in_dim %4820, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4822 = stablehlo.divide %4210, %4821 : tensor<576xbf16>
    %4823 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4824 = stablehlo.broadcast_in_dim %4823, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4825 = stablehlo.divide %4215, %4824 : tensor<576x9x128xbf16>
    %4826 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4827 = stablehlo.broadcast_in_dim %4826, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4828 = stablehlo.divide %4220, %4827 : tensor<9x128x576xbf16>
    %4829 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4830 = stablehlo.broadcast_in_dim %4829, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4831 = stablehlo.divide %4225, %4830 : tensor<576x9x128xbf16>
    %4832 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4833 = stablehlo.broadcast_in_dim %4832, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4834 = stablehlo.divide %4230, %4833 : tensor<576x9x128xbf16>
    %4835 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4836 = stablehlo.broadcast_in_dim %4835, dims = [] : (tensor<bf16>) -> tensor<576x128256xbf16>
    %4837 = stablehlo.divide %4235, %4836 : tensor<576x128256xbf16>
    %4838 = stablehlo.convert %4642 : (tensor<f32>) -> tensor<bf16>
    %4839 = stablehlo.broadcast_in_dim %4838, dims = [] : (tensor<bf16>) -> tensor<128256x576xbf16>
    %4840 = stablehlo.divide %4240, %4839 : tensor<128256x576xbf16>
    %cst_857 = stablehlo.constant dense<0.949999988> : tensor<f32>
    %4841 = stablehlo.convert %4639 : (tensor<i32>) -> tensor<f32>
    %4842 = stablehlo.power %cst_857, %4841 : tensor<f32>
    %cst_858 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4843 = stablehlo.subtract %cst_858, %4842 : tensor<f32>
    %4844 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4845 = stablehlo.broadcast_in_dim %4844, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4846 = stablehlo.divide %4246, %4845 : tensor<576xbf16>
    %4847 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4848 = stablehlo.broadcast_in_dim %4847, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4849 = stablehlo.divide %4252, %4848 : tensor<576x2016xbf16>
    %4850 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4851 = stablehlo.broadcast_in_dim %4850, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4852 = stablehlo.divide %4258, %4851 : tensor<576x2016xbf16>
    %4853 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4854 = stablehlo.broadcast_in_dim %4853, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4855 = stablehlo.divide %4264, %4854 : tensor<2016x576xbf16>
    %4856 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4857 = stablehlo.broadcast_in_dim %4856, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4858 = stablehlo.divide %4270, %4857 : tensor<576xbf16>
    %4859 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4860 = stablehlo.broadcast_in_dim %4859, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4861 = stablehlo.divide %4276, %4860 : tensor<576xbf16>
    %4862 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4863 = stablehlo.broadcast_in_dim %4862, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4864 = stablehlo.divide %4282, %4863 : tensor<576x9x128xbf16>
    %4865 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4866 = stablehlo.broadcast_in_dim %4865, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4867 = stablehlo.divide %4288, %4866 : tensor<9x128x576xbf16>
    %4868 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4869 = stablehlo.broadcast_in_dim %4868, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4870 = stablehlo.divide %4294, %4869 : tensor<576x9x128xbf16>
    %4871 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4872 = stablehlo.broadcast_in_dim %4871, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4873 = stablehlo.divide %4300, %4872 : tensor<576x9x128xbf16>
    %4874 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4875 = stablehlo.broadcast_in_dim %4874, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4876 = stablehlo.divide %4306, %4875 : tensor<576x2016xbf16>
    %4877 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4878 = stablehlo.broadcast_in_dim %4877, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4879 = stablehlo.divide %4312, %4878 : tensor<576x2016xbf16>
    %4880 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4881 = stablehlo.broadcast_in_dim %4880, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4882 = stablehlo.divide %4318, %4881 : tensor<2016x576xbf16>
    %4883 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4884 = stablehlo.broadcast_in_dim %4883, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4885 = stablehlo.divide %4324, %4884 : tensor<576xbf16>
    %4886 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4887 = stablehlo.broadcast_in_dim %4886, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4888 = stablehlo.divide %4330, %4887 : tensor<576xbf16>
    %4889 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4890 = stablehlo.broadcast_in_dim %4889, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4891 = stablehlo.divide %4336, %4890 : tensor<576x9x128xbf16>
    %4892 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4893 = stablehlo.broadcast_in_dim %4892, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4894 = stablehlo.divide %4342, %4893 : tensor<9x128x576xbf16>
    %4895 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4896 = stablehlo.broadcast_in_dim %4895, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4897 = stablehlo.divide %4348, %4896 : tensor<576x9x128xbf16>
    %4898 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4899 = stablehlo.broadcast_in_dim %4898, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4900 = stablehlo.divide %4354, %4899 : tensor<576x9x128xbf16>
    %4901 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4902 = stablehlo.broadcast_in_dim %4901, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4903 = stablehlo.divide %4360, %4902 : tensor<576x2016xbf16>
    %4904 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4905 = stablehlo.broadcast_in_dim %4904, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4906 = stablehlo.divide %4366, %4905 : tensor<576x2016xbf16>
    %4907 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4908 = stablehlo.broadcast_in_dim %4907, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4909 = stablehlo.divide %4372, %4908 : tensor<2016x576xbf16>
    %4910 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4911 = stablehlo.broadcast_in_dim %4910, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4912 = stablehlo.divide %4378, %4911 : tensor<576xbf16>
    %4913 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4914 = stablehlo.broadcast_in_dim %4913, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4915 = stablehlo.divide %4384, %4914 : tensor<576xbf16>
    %4916 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4917 = stablehlo.broadcast_in_dim %4916, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4918 = stablehlo.divide %4390, %4917 : tensor<576x9x128xbf16>
    %4919 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4920 = stablehlo.broadcast_in_dim %4919, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4921 = stablehlo.divide %4396, %4920 : tensor<9x128x576xbf16>
    %4922 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4923 = stablehlo.broadcast_in_dim %4922, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4924 = stablehlo.divide %4402, %4923 : tensor<576x9x128xbf16>
    %4925 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4926 = stablehlo.broadcast_in_dim %4925, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4927 = stablehlo.divide %4408, %4926 : tensor<576x9x128xbf16>
    %4928 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4929 = stablehlo.broadcast_in_dim %4928, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4930 = stablehlo.divide %4414, %4929 : tensor<576x2016xbf16>
    %4931 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4932 = stablehlo.broadcast_in_dim %4931, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4933 = stablehlo.divide %4420, %4932 : tensor<576x2016xbf16>
    %4934 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4935 = stablehlo.broadcast_in_dim %4934, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4936 = stablehlo.divide %4426, %4935 : tensor<2016x576xbf16>
    %4937 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4938 = stablehlo.broadcast_in_dim %4937, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4939 = stablehlo.divide %4432, %4938 : tensor<576xbf16>
    %4940 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4941 = stablehlo.broadcast_in_dim %4940, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4942 = stablehlo.divide %4438, %4941 : tensor<576xbf16>
    %4943 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4944 = stablehlo.broadcast_in_dim %4943, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4945 = stablehlo.divide %4444, %4944 : tensor<576x9x128xbf16>
    %4946 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4947 = stablehlo.broadcast_in_dim %4946, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4948 = stablehlo.divide %4450, %4947 : tensor<9x128x576xbf16>
    %4949 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4950 = stablehlo.broadcast_in_dim %4949, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4951 = stablehlo.divide %4456, %4950 : tensor<576x9x128xbf16>
    %4952 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4953 = stablehlo.broadcast_in_dim %4952, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4954 = stablehlo.divide %4462, %4953 : tensor<576x9x128xbf16>
    %4955 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4956 = stablehlo.broadcast_in_dim %4955, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4957 = stablehlo.divide %4468, %4956 : tensor<576x2016xbf16>
    %4958 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4959 = stablehlo.broadcast_in_dim %4958, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4960 = stablehlo.divide %4474, %4959 : tensor<576x2016xbf16>
    %4961 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4962 = stablehlo.broadcast_in_dim %4961, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4963 = stablehlo.divide %4480, %4962 : tensor<2016x576xbf16>
    %4964 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4965 = stablehlo.broadcast_in_dim %4964, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4966 = stablehlo.divide %4486, %4965 : tensor<576xbf16>
    %4967 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4968 = stablehlo.broadcast_in_dim %4967, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4969 = stablehlo.divide %4492, %4968 : tensor<576xbf16>
    %4970 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4971 = stablehlo.broadcast_in_dim %4970, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4972 = stablehlo.divide %4498, %4971 : tensor<576x9x128xbf16>
    %4973 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4974 = stablehlo.broadcast_in_dim %4973, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %4975 = stablehlo.divide %4504, %4974 : tensor<9x128x576xbf16>
    %4976 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4977 = stablehlo.broadcast_in_dim %4976, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4978 = stablehlo.divide %4510, %4977 : tensor<576x9x128xbf16>
    %4979 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4980 = stablehlo.broadcast_in_dim %4979, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4981 = stablehlo.divide %4516, %4980 : tensor<576x9x128xbf16>
    %4982 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4983 = stablehlo.broadcast_in_dim %4982, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4984 = stablehlo.divide %4522, %4983 : tensor<576x2016xbf16>
    %4985 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4986 = stablehlo.broadcast_in_dim %4985, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %4987 = stablehlo.divide %4528, %4986 : tensor<576x2016xbf16>
    %4988 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4989 = stablehlo.broadcast_in_dim %4988, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %4990 = stablehlo.divide %4534, %4989 : tensor<2016x576xbf16>
    %4991 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4992 = stablehlo.broadcast_in_dim %4991, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4993 = stablehlo.divide %4540, %4992 : tensor<576xbf16>
    %4994 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4995 = stablehlo.broadcast_in_dim %4994, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %4996 = stablehlo.divide %4546, %4995 : tensor<576xbf16>
    %4997 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4998 = stablehlo.broadcast_in_dim %4997, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %4999 = stablehlo.divide %4552, %4998 : tensor<576x9x128xbf16>
    %5000 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5001 = stablehlo.broadcast_in_dim %5000, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5002 = stablehlo.divide %4558, %5001 : tensor<9x128x576xbf16>
    %5003 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5004 = stablehlo.broadcast_in_dim %5003, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5005 = stablehlo.divide %4564, %5004 : tensor<576x9x128xbf16>
    %5006 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5007 = stablehlo.broadcast_in_dim %5006, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5008 = stablehlo.divide %4570, %5007 : tensor<576x9x128xbf16>
    %5009 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5010 = stablehlo.broadcast_in_dim %5009, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5011 = stablehlo.divide %4576, %5010 : tensor<576x2016xbf16>
    %5012 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5013 = stablehlo.broadcast_in_dim %5012, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5014 = stablehlo.divide %4582, %5013 : tensor<576x2016xbf16>
    %5015 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5016 = stablehlo.broadcast_in_dim %5015, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5017 = stablehlo.divide %4588, %5016 : tensor<2016x576xbf16>
    %5018 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5019 = stablehlo.broadcast_in_dim %5018, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5020 = stablehlo.divide %4594, %5019 : tensor<576xbf16>
    %5021 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5022 = stablehlo.broadcast_in_dim %5021, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5023 = stablehlo.divide %4600, %5022 : tensor<576xbf16>
    %5024 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5025 = stablehlo.broadcast_in_dim %5024, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5026 = stablehlo.divide %4606, %5025 : tensor<576x9x128xbf16>
    %5027 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5028 = stablehlo.broadcast_in_dim %5027, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5029 = stablehlo.divide %4612, %5028 : tensor<9x128x576xbf16>
    %5030 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5031 = stablehlo.broadcast_in_dim %5030, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5032 = stablehlo.divide %4618, %5031 : tensor<576x9x128xbf16>
    %5033 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5034 = stablehlo.broadcast_in_dim %5033, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5035 = stablehlo.divide %4624, %5034 : tensor<576x9x128xbf16>
    %5036 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5037 = stablehlo.broadcast_in_dim %5036, dims = [] : (tensor<bf16>) -> tensor<576x128256xbf16>
    %5038 = stablehlo.divide %4630, %5037 : tensor<576x128256xbf16>
    %5039 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %5040 = stablehlo.broadcast_in_dim %5039, dims = [] : (tensor<bf16>) -> tensor<128256x576xbf16>
    %5041 = stablehlo.divide %4636, %5040 : tensor<128256x576xbf16>
    %cst_859 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5042 = stablehlo.broadcast_in_dim %cst_859, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5043 = stablehlo.add %4846, %5042 : tensor<576xbf16>
    %5044 = stablehlo.sqrt %5043 : tensor<576xbf16>
    %cst_860 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5045 = stablehlo.broadcast_in_dim %cst_860, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5046 = stablehlo.add %5044, %5045 : tensor<576xbf16>
    %5047 = stablehlo.divide %4645, %5046 : tensor<576xbf16>
    %cst_861 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5048 = stablehlo.broadcast_in_dim %cst_861, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5049 = stablehlo.add %4849, %5048 : tensor<576x2016xbf16>
    %5050 = stablehlo.sqrt %5049 : tensor<576x2016xbf16>
    %cst_862 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5051 = stablehlo.broadcast_in_dim %cst_862, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5052 = stablehlo.add %5050, %5051 : tensor<576x2016xbf16>
    %5053 = stablehlo.divide %4648, %5052 : tensor<576x2016xbf16>
    %cst_863 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5054 = stablehlo.broadcast_in_dim %cst_863, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5055 = stablehlo.add %4852, %5054 : tensor<576x2016xbf16>
    %5056 = stablehlo.sqrt %5055 : tensor<576x2016xbf16>
    %cst_864 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5057 = stablehlo.broadcast_in_dim %cst_864, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5058 = stablehlo.add %5056, %5057 : tensor<576x2016xbf16>
    %5059 = stablehlo.divide %4651, %5058 : tensor<576x2016xbf16>
    %cst_865 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5060 = stablehlo.broadcast_in_dim %cst_865, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5061 = stablehlo.add %4855, %5060 : tensor<2016x576xbf16>
    %5062 = stablehlo.sqrt %5061 : tensor<2016x576xbf16>
    %cst_866 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5063 = stablehlo.broadcast_in_dim %cst_866, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5064 = stablehlo.add %5062, %5063 : tensor<2016x576xbf16>
    %5065 = stablehlo.divide %4654, %5064 : tensor<2016x576xbf16>
    %cst_867 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5066 = stablehlo.broadcast_in_dim %cst_867, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5067 = stablehlo.add %4858, %5066 : tensor<576xbf16>
    %5068 = stablehlo.sqrt %5067 : tensor<576xbf16>
    %cst_868 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5069 = stablehlo.broadcast_in_dim %cst_868, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5070 = stablehlo.add %5068, %5069 : tensor<576xbf16>
    %5071 = stablehlo.divide %4657, %5070 : tensor<576xbf16>
    %cst_869 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5072 = stablehlo.broadcast_in_dim %cst_869, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5073 = stablehlo.add %4861, %5072 : tensor<576xbf16>
    %5074 = stablehlo.sqrt %5073 : tensor<576xbf16>
    %cst_870 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5075 = stablehlo.broadcast_in_dim %cst_870, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5076 = stablehlo.add %5074, %5075 : tensor<576xbf16>
    %5077 = stablehlo.divide %4660, %5076 : tensor<576xbf16>
    %cst_871 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5078 = stablehlo.broadcast_in_dim %cst_871, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5079 = stablehlo.add %4864, %5078 : tensor<576x9x128xbf16>
    %5080 = stablehlo.sqrt %5079 : tensor<576x9x128xbf16>
    %cst_872 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5081 = stablehlo.broadcast_in_dim %cst_872, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5082 = stablehlo.add %5080, %5081 : tensor<576x9x128xbf16>
    %5083 = stablehlo.divide %4663, %5082 : tensor<576x9x128xbf16>
    %cst_873 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5084 = stablehlo.broadcast_in_dim %cst_873, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5085 = stablehlo.add %4867, %5084 : tensor<9x128x576xbf16>
    %5086 = stablehlo.sqrt %5085 : tensor<9x128x576xbf16>
    %cst_874 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5087 = stablehlo.broadcast_in_dim %cst_874, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5088 = stablehlo.add %5086, %5087 : tensor<9x128x576xbf16>
    %5089 = stablehlo.divide %4666, %5088 : tensor<9x128x576xbf16>
    %cst_875 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5090 = stablehlo.broadcast_in_dim %cst_875, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5091 = stablehlo.add %4870, %5090 : tensor<576x9x128xbf16>
    %5092 = stablehlo.sqrt %5091 : tensor<576x9x128xbf16>
    %cst_876 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5093 = stablehlo.broadcast_in_dim %cst_876, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5094 = stablehlo.add %5092, %5093 : tensor<576x9x128xbf16>
    %5095 = stablehlo.divide %4669, %5094 : tensor<576x9x128xbf16>
    %cst_877 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5096 = stablehlo.broadcast_in_dim %cst_877, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5097 = stablehlo.add %4873, %5096 : tensor<576x9x128xbf16>
    %5098 = stablehlo.sqrt %5097 : tensor<576x9x128xbf16>
    %cst_878 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5099 = stablehlo.broadcast_in_dim %cst_878, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5100 = stablehlo.add %5098, %5099 : tensor<576x9x128xbf16>
    %5101 = stablehlo.divide %4672, %5100 : tensor<576x9x128xbf16>
    %cst_879 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5102 = stablehlo.broadcast_in_dim %cst_879, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5103 = stablehlo.add %4876, %5102 : tensor<576x2016xbf16>
    %5104 = stablehlo.sqrt %5103 : tensor<576x2016xbf16>
    %cst_880 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5105 = stablehlo.broadcast_in_dim %cst_880, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5106 = stablehlo.add %5104, %5105 : tensor<576x2016xbf16>
    %5107 = stablehlo.divide %4675, %5106 : tensor<576x2016xbf16>
    %cst_881 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5108 = stablehlo.broadcast_in_dim %cst_881, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5109 = stablehlo.add %4879, %5108 : tensor<576x2016xbf16>
    %5110 = stablehlo.sqrt %5109 : tensor<576x2016xbf16>
    %cst_882 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5111 = stablehlo.broadcast_in_dim %cst_882, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5112 = stablehlo.add %5110, %5111 : tensor<576x2016xbf16>
    %5113 = stablehlo.divide %4678, %5112 : tensor<576x2016xbf16>
    %cst_883 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5114 = stablehlo.broadcast_in_dim %cst_883, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5115 = stablehlo.add %4882, %5114 : tensor<2016x576xbf16>
    %5116 = stablehlo.sqrt %5115 : tensor<2016x576xbf16>
    %cst_884 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5117 = stablehlo.broadcast_in_dim %cst_884, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5118 = stablehlo.add %5116, %5117 : tensor<2016x576xbf16>
    %5119 = stablehlo.divide %4681, %5118 : tensor<2016x576xbf16>
    %cst_885 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5120 = stablehlo.broadcast_in_dim %cst_885, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5121 = stablehlo.add %4885, %5120 : tensor<576xbf16>
    %5122 = stablehlo.sqrt %5121 : tensor<576xbf16>
    %cst_886 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5123 = stablehlo.broadcast_in_dim %cst_886, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5124 = stablehlo.add %5122, %5123 : tensor<576xbf16>
    %5125 = stablehlo.divide %4684, %5124 : tensor<576xbf16>
    %cst_887 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5126 = stablehlo.broadcast_in_dim %cst_887, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5127 = stablehlo.add %4888, %5126 : tensor<576xbf16>
    %5128 = stablehlo.sqrt %5127 : tensor<576xbf16>
    %cst_888 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5129 = stablehlo.broadcast_in_dim %cst_888, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5130 = stablehlo.add %5128, %5129 : tensor<576xbf16>
    %5131 = stablehlo.divide %4687, %5130 : tensor<576xbf16>
    %cst_889 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5132 = stablehlo.broadcast_in_dim %cst_889, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5133 = stablehlo.add %4891, %5132 : tensor<576x9x128xbf16>
    %5134 = stablehlo.sqrt %5133 : tensor<576x9x128xbf16>
    %cst_890 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5135 = stablehlo.broadcast_in_dim %cst_890, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5136 = stablehlo.add %5134, %5135 : tensor<576x9x128xbf16>
    %5137 = stablehlo.divide %4690, %5136 : tensor<576x9x128xbf16>
    %cst_891 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5138 = stablehlo.broadcast_in_dim %cst_891, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5139 = stablehlo.add %4894, %5138 : tensor<9x128x576xbf16>
    %5140 = stablehlo.sqrt %5139 : tensor<9x128x576xbf16>
    %cst_892 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5141 = stablehlo.broadcast_in_dim %cst_892, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5142 = stablehlo.add %5140, %5141 : tensor<9x128x576xbf16>
    %5143 = stablehlo.divide %4693, %5142 : tensor<9x128x576xbf16>
    %cst_893 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5144 = stablehlo.broadcast_in_dim %cst_893, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5145 = stablehlo.add %4897, %5144 : tensor<576x9x128xbf16>
    %5146 = stablehlo.sqrt %5145 : tensor<576x9x128xbf16>
    %cst_894 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5147 = stablehlo.broadcast_in_dim %cst_894, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5148 = stablehlo.add %5146, %5147 : tensor<576x9x128xbf16>
    %5149 = stablehlo.divide %4696, %5148 : tensor<576x9x128xbf16>
    %cst_895 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5150 = stablehlo.broadcast_in_dim %cst_895, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5151 = stablehlo.add %4900, %5150 : tensor<576x9x128xbf16>
    %5152 = stablehlo.sqrt %5151 : tensor<576x9x128xbf16>
    %cst_896 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5153 = stablehlo.broadcast_in_dim %cst_896, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5154 = stablehlo.add %5152, %5153 : tensor<576x9x128xbf16>
    %5155 = stablehlo.divide %4699, %5154 : tensor<576x9x128xbf16>
    %cst_897 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5156 = stablehlo.broadcast_in_dim %cst_897, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5157 = stablehlo.add %4903, %5156 : tensor<576x2016xbf16>
    %5158 = stablehlo.sqrt %5157 : tensor<576x2016xbf16>
    %cst_898 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5159 = stablehlo.broadcast_in_dim %cst_898, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5160 = stablehlo.add %5158, %5159 : tensor<576x2016xbf16>
    %5161 = stablehlo.divide %4702, %5160 : tensor<576x2016xbf16>
    %cst_899 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5162 = stablehlo.broadcast_in_dim %cst_899, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5163 = stablehlo.add %4906, %5162 : tensor<576x2016xbf16>
    %5164 = stablehlo.sqrt %5163 : tensor<576x2016xbf16>
    %cst_900 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5165 = stablehlo.broadcast_in_dim %cst_900, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5166 = stablehlo.add %5164, %5165 : tensor<576x2016xbf16>
    %5167 = stablehlo.divide %4705, %5166 : tensor<576x2016xbf16>
    %cst_901 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5168 = stablehlo.broadcast_in_dim %cst_901, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5169 = stablehlo.add %4909, %5168 : tensor<2016x576xbf16>
    %5170 = stablehlo.sqrt %5169 : tensor<2016x576xbf16>
    %cst_902 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5171 = stablehlo.broadcast_in_dim %cst_902, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5172 = stablehlo.add %5170, %5171 : tensor<2016x576xbf16>
    %5173 = stablehlo.divide %4708, %5172 : tensor<2016x576xbf16>
    %cst_903 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5174 = stablehlo.broadcast_in_dim %cst_903, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5175 = stablehlo.add %4912, %5174 : tensor<576xbf16>
    %5176 = stablehlo.sqrt %5175 : tensor<576xbf16>
    %cst_904 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5177 = stablehlo.broadcast_in_dim %cst_904, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5178 = stablehlo.add %5176, %5177 : tensor<576xbf16>
    %5179 = stablehlo.divide %4711, %5178 : tensor<576xbf16>
    %cst_905 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5180 = stablehlo.broadcast_in_dim %cst_905, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5181 = stablehlo.add %4915, %5180 : tensor<576xbf16>
    %5182 = stablehlo.sqrt %5181 : tensor<576xbf16>
    %cst_906 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5183 = stablehlo.broadcast_in_dim %cst_906, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5184 = stablehlo.add %5182, %5183 : tensor<576xbf16>
    %5185 = stablehlo.divide %4714, %5184 : tensor<576xbf16>
    %cst_907 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5186 = stablehlo.broadcast_in_dim %cst_907, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5187 = stablehlo.add %4918, %5186 : tensor<576x9x128xbf16>
    %5188 = stablehlo.sqrt %5187 : tensor<576x9x128xbf16>
    %cst_908 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5189 = stablehlo.broadcast_in_dim %cst_908, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5190 = stablehlo.add %5188, %5189 : tensor<576x9x128xbf16>
    %5191 = stablehlo.divide %4717, %5190 : tensor<576x9x128xbf16>
    %cst_909 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5192 = stablehlo.broadcast_in_dim %cst_909, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5193 = stablehlo.add %4921, %5192 : tensor<9x128x576xbf16>
    %5194 = stablehlo.sqrt %5193 : tensor<9x128x576xbf16>
    %cst_910 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5195 = stablehlo.broadcast_in_dim %cst_910, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5196 = stablehlo.add %5194, %5195 : tensor<9x128x576xbf16>
    %5197 = stablehlo.divide %4720, %5196 : tensor<9x128x576xbf16>
    %cst_911 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5198 = stablehlo.broadcast_in_dim %cst_911, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5199 = stablehlo.add %4924, %5198 : tensor<576x9x128xbf16>
    %5200 = stablehlo.sqrt %5199 : tensor<576x9x128xbf16>
    %cst_912 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5201 = stablehlo.broadcast_in_dim %cst_912, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5202 = stablehlo.add %5200, %5201 : tensor<576x9x128xbf16>
    %5203 = stablehlo.divide %4723, %5202 : tensor<576x9x128xbf16>
    %cst_913 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5204 = stablehlo.broadcast_in_dim %cst_913, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5205 = stablehlo.add %4927, %5204 : tensor<576x9x128xbf16>
    %5206 = stablehlo.sqrt %5205 : tensor<576x9x128xbf16>
    %cst_914 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5207 = stablehlo.broadcast_in_dim %cst_914, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5208 = stablehlo.add %5206, %5207 : tensor<576x9x128xbf16>
    %5209 = stablehlo.divide %4726, %5208 : tensor<576x9x128xbf16>
    %cst_915 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5210 = stablehlo.broadcast_in_dim %cst_915, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5211 = stablehlo.add %4930, %5210 : tensor<576x2016xbf16>
    %5212 = stablehlo.sqrt %5211 : tensor<576x2016xbf16>
    %cst_916 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5213 = stablehlo.broadcast_in_dim %cst_916, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5214 = stablehlo.add %5212, %5213 : tensor<576x2016xbf16>
    %5215 = stablehlo.divide %4729, %5214 : tensor<576x2016xbf16>
    %cst_917 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5216 = stablehlo.broadcast_in_dim %cst_917, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5217 = stablehlo.add %4933, %5216 : tensor<576x2016xbf16>
    %5218 = stablehlo.sqrt %5217 : tensor<576x2016xbf16>
    %cst_918 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5219 = stablehlo.broadcast_in_dim %cst_918, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5220 = stablehlo.add %5218, %5219 : tensor<576x2016xbf16>
    %5221 = stablehlo.divide %4732, %5220 : tensor<576x2016xbf16>
    %cst_919 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5222 = stablehlo.broadcast_in_dim %cst_919, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5223 = stablehlo.add %4936, %5222 : tensor<2016x576xbf16>
    %5224 = stablehlo.sqrt %5223 : tensor<2016x576xbf16>
    %cst_920 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5225 = stablehlo.broadcast_in_dim %cst_920, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5226 = stablehlo.add %5224, %5225 : tensor<2016x576xbf16>
    %5227 = stablehlo.divide %4735, %5226 : tensor<2016x576xbf16>
    %cst_921 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5228 = stablehlo.broadcast_in_dim %cst_921, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5229 = stablehlo.add %4939, %5228 : tensor<576xbf16>
    %5230 = stablehlo.sqrt %5229 : tensor<576xbf16>
    %cst_922 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5231 = stablehlo.broadcast_in_dim %cst_922, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5232 = stablehlo.add %5230, %5231 : tensor<576xbf16>
    %5233 = stablehlo.divide %4738, %5232 : tensor<576xbf16>
    %cst_923 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5234 = stablehlo.broadcast_in_dim %cst_923, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5235 = stablehlo.add %4942, %5234 : tensor<576xbf16>
    %5236 = stablehlo.sqrt %5235 : tensor<576xbf16>
    %cst_924 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5237 = stablehlo.broadcast_in_dim %cst_924, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5238 = stablehlo.add %5236, %5237 : tensor<576xbf16>
    %5239 = stablehlo.divide %4741, %5238 : tensor<576xbf16>
    %cst_925 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5240 = stablehlo.broadcast_in_dim %cst_925, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5241 = stablehlo.add %4945, %5240 : tensor<576x9x128xbf16>
    %5242 = stablehlo.sqrt %5241 : tensor<576x9x128xbf16>
    %cst_926 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5243 = stablehlo.broadcast_in_dim %cst_926, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5244 = stablehlo.add %5242, %5243 : tensor<576x9x128xbf16>
    %5245 = stablehlo.divide %4744, %5244 : tensor<576x9x128xbf16>
    %cst_927 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5246 = stablehlo.broadcast_in_dim %cst_927, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5247 = stablehlo.add %4948, %5246 : tensor<9x128x576xbf16>
    %5248 = stablehlo.sqrt %5247 : tensor<9x128x576xbf16>
    %cst_928 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5249 = stablehlo.broadcast_in_dim %cst_928, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5250 = stablehlo.add %5248, %5249 : tensor<9x128x576xbf16>
    %5251 = stablehlo.divide %4747, %5250 : tensor<9x128x576xbf16>
    %cst_929 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5252 = stablehlo.broadcast_in_dim %cst_929, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5253 = stablehlo.add %4951, %5252 : tensor<576x9x128xbf16>
    %5254 = stablehlo.sqrt %5253 : tensor<576x9x128xbf16>
    %cst_930 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5255 = stablehlo.broadcast_in_dim %cst_930, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5256 = stablehlo.add %5254, %5255 : tensor<576x9x128xbf16>
    %5257 = stablehlo.divide %4750, %5256 : tensor<576x9x128xbf16>
    %cst_931 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5258 = stablehlo.broadcast_in_dim %cst_931, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5259 = stablehlo.add %4954, %5258 : tensor<576x9x128xbf16>
    %5260 = stablehlo.sqrt %5259 : tensor<576x9x128xbf16>
    %cst_932 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5261 = stablehlo.broadcast_in_dim %cst_932, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5262 = stablehlo.add %5260, %5261 : tensor<576x9x128xbf16>
    %5263 = stablehlo.divide %4753, %5262 : tensor<576x9x128xbf16>
    %cst_933 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5264 = stablehlo.broadcast_in_dim %cst_933, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5265 = stablehlo.add %4957, %5264 : tensor<576x2016xbf16>
    %5266 = stablehlo.sqrt %5265 : tensor<576x2016xbf16>
    %cst_934 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5267 = stablehlo.broadcast_in_dim %cst_934, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5268 = stablehlo.add %5266, %5267 : tensor<576x2016xbf16>
    %5269 = stablehlo.divide %4756, %5268 : tensor<576x2016xbf16>
    %cst_935 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5270 = stablehlo.broadcast_in_dim %cst_935, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5271 = stablehlo.add %4960, %5270 : tensor<576x2016xbf16>
    %5272 = stablehlo.sqrt %5271 : tensor<576x2016xbf16>
    %cst_936 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5273 = stablehlo.broadcast_in_dim %cst_936, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5274 = stablehlo.add %5272, %5273 : tensor<576x2016xbf16>
    %5275 = stablehlo.divide %4759, %5274 : tensor<576x2016xbf16>
    %cst_937 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5276 = stablehlo.broadcast_in_dim %cst_937, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5277 = stablehlo.add %4963, %5276 : tensor<2016x576xbf16>
    %5278 = stablehlo.sqrt %5277 : tensor<2016x576xbf16>
    %cst_938 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5279 = stablehlo.broadcast_in_dim %cst_938, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5280 = stablehlo.add %5278, %5279 : tensor<2016x576xbf16>
    %5281 = stablehlo.divide %4762, %5280 : tensor<2016x576xbf16>
    %cst_939 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5282 = stablehlo.broadcast_in_dim %cst_939, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5283 = stablehlo.add %4966, %5282 : tensor<576xbf16>
    %5284 = stablehlo.sqrt %5283 : tensor<576xbf16>
    %cst_940 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5285 = stablehlo.broadcast_in_dim %cst_940, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5286 = stablehlo.add %5284, %5285 : tensor<576xbf16>
    %5287 = stablehlo.divide %4765, %5286 : tensor<576xbf16>
    %cst_941 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5288 = stablehlo.broadcast_in_dim %cst_941, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5289 = stablehlo.add %4969, %5288 : tensor<576xbf16>
    %5290 = stablehlo.sqrt %5289 : tensor<576xbf16>
    %cst_942 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5291 = stablehlo.broadcast_in_dim %cst_942, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5292 = stablehlo.add %5290, %5291 : tensor<576xbf16>
    %5293 = stablehlo.divide %4768, %5292 : tensor<576xbf16>
    %cst_943 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5294 = stablehlo.broadcast_in_dim %cst_943, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5295 = stablehlo.add %4972, %5294 : tensor<576x9x128xbf16>
    %5296 = stablehlo.sqrt %5295 : tensor<576x9x128xbf16>
    %cst_944 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5297 = stablehlo.broadcast_in_dim %cst_944, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5298 = stablehlo.add %5296, %5297 : tensor<576x9x128xbf16>
    %5299 = stablehlo.divide %4771, %5298 : tensor<576x9x128xbf16>
    %cst_945 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5300 = stablehlo.broadcast_in_dim %cst_945, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5301 = stablehlo.add %4975, %5300 : tensor<9x128x576xbf16>
    %5302 = stablehlo.sqrt %5301 : tensor<9x128x576xbf16>
    %cst_946 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5303 = stablehlo.broadcast_in_dim %cst_946, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5304 = stablehlo.add %5302, %5303 : tensor<9x128x576xbf16>
    %5305 = stablehlo.divide %4774, %5304 : tensor<9x128x576xbf16>
    %cst_947 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5306 = stablehlo.broadcast_in_dim %cst_947, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5307 = stablehlo.add %4978, %5306 : tensor<576x9x128xbf16>
    %5308 = stablehlo.sqrt %5307 : tensor<576x9x128xbf16>
    %cst_948 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5309 = stablehlo.broadcast_in_dim %cst_948, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5310 = stablehlo.add %5308, %5309 : tensor<576x9x128xbf16>
    %5311 = stablehlo.divide %4777, %5310 : tensor<576x9x128xbf16>
    %cst_949 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5312 = stablehlo.broadcast_in_dim %cst_949, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5313 = stablehlo.add %4981, %5312 : tensor<576x9x128xbf16>
    %5314 = stablehlo.sqrt %5313 : tensor<576x9x128xbf16>
    %cst_950 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5315 = stablehlo.broadcast_in_dim %cst_950, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5316 = stablehlo.add %5314, %5315 : tensor<576x9x128xbf16>
    %5317 = stablehlo.divide %4780, %5316 : tensor<576x9x128xbf16>
    %cst_951 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5318 = stablehlo.broadcast_in_dim %cst_951, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5319 = stablehlo.add %4984, %5318 : tensor<576x2016xbf16>
    %5320 = stablehlo.sqrt %5319 : tensor<576x2016xbf16>
    %cst_952 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5321 = stablehlo.broadcast_in_dim %cst_952, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5322 = stablehlo.add %5320, %5321 : tensor<576x2016xbf16>
    %5323 = stablehlo.divide %4783, %5322 : tensor<576x2016xbf16>
    %cst_953 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5324 = stablehlo.broadcast_in_dim %cst_953, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5325 = stablehlo.add %4987, %5324 : tensor<576x2016xbf16>
    %5326 = stablehlo.sqrt %5325 : tensor<576x2016xbf16>
    %cst_954 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5327 = stablehlo.broadcast_in_dim %cst_954, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5328 = stablehlo.add %5326, %5327 : tensor<576x2016xbf16>
    %5329 = stablehlo.divide %4786, %5328 : tensor<576x2016xbf16>
    %cst_955 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5330 = stablehlo.broadcast_in_dim %cst_955, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5331 = stablehlo.add %4990, %5330 : tensor<2016x576xbf16>
    %5332 = stablehlo.sqrt %5331 : tensor<2016x576xbf16>
    %cst_956 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5333 = stablehlo.broadcast_in_dim %cst_956, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5334 = stablehlo.add %5332, %5333 : tensor<2016x576xbf16>
    %5335 = stablehlo.divide %4789, %5334 : tensor<2016x576xbf16>
    %cst_957 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5336 = stablehlo.broadcast_in_dim %cst_957, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5337 = stablehlo.add %4993, %5336 : tensor<576xbf16>
    %5338 = stablehlo.sqrt %5337 : tensor<576xbf16>
    %cst_958 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5339 = stablehlo.broadcast_in_dim %cst_958, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5340 = stablehlo.add %5338, %5339 : tensor<576xbf16>
    %5341 = stablehlo.divide %4792, %5340 : tensor<576xbf16>
    %cst_959 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5342 = stablehlo.broadcast_in_dim %cst_959, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5343 = stablehlo.add %4996, %5342 : tensor<576xbf16>
    %5344 = stablehlo.sqrt %5343 : tensor<576xbf16>
    %cst_960 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5345 = stablehlo.broadcast_in_dim %cst_960, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5346 = stablehlo.add %5344, %5345 : tensor<576xbf16>
    %5347 = stablehlo.divide %4795, %5346 : tensor<576xbf16>
    %cst_961 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5348 = stablehlo.broadcast_in_dim %cst_961, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5349 = stablehlo.add %4999, %5348 : tensor<576x9x128xbf16>
    %5350 = stablehlo.sqrt %5349 : tensor<576x9x128xbf16>
    %cst_962 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5351 = stablehlo.broadcast_in_dim %cst_962, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5352 = stablehlo.add %5350, %5351 : tensor<576x9x128xbf16>
    %5353 = stablehlo.divide %4798, %5352 : tensor<576x9x128xbf16>
    %cst_963 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5354 = stablehlo.broadcast_in_dim %cst_963, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5355 = stablehlo.add %5002, %5354 : tensor<9x128x576xbf16>
    %5356 = stablehlo.sqrt %5355 : tensor<9x128x576xbf16>
    %cst_964 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5357 = stablehlo.broadcast_in_dim %cst_964, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5358 = stablehlo.add %5356, %5357 : tensor<9x128x576xbf16>
    %5359 = stablehlo.divide %4801, %5358 : tensor<9x128x576xbf16>
    %cst_965 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5360 = stablehlo.broadcast_in_dim %cst_965, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5361 = stablehlo.add %5005, %5360 : tensor<576x9x128xbf16>
    %5362 = stablehlo.sqrt %5361 : tensor<576x9x128xbf16>
    %cst_966 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5363 = stablehlo.broadcast_in_dim %cst_966, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5364 = stablehlo.add %5362, %5363 : tensor<576x9x128xbf16>
    %5365 = stablehlo.divide %4804, %5364 : tensor<576x9x128xbf16>
    %cst_967 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5366 = stablehlo.broadcast_in_dim %cst_967, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5367 = stablehlo.add %5008, %5366 : tensor<576x9x128xbf16>
    %5368 = stablehlo.sqrt %5367 : tensor<576x9x128xbf16>
    %cst_968 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5369 = stablehlo.broadcast_in_dim %cst_968, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5370 = stablehlo.add %5368, %5369 : tensor<576x9x128xbf16>
    %5371 = stablehlo.divide %4807, %5370 : tensor<576x9x128xbf16>
    %cst_969 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5372 = stablehlo.broadcast_in_dim %cst_969, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5373 = stablehlo.add %5011, %5372 : tensor<576x2016xbf16>
    %5374 = stablehlo.sqrt %5373 : tensor<576x2016xbf16>
    %cst_970 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5375 = stablehlo.broadcast_in_dim %cst_970, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5376 = stablehlo.add %5374, %5375 : tensor<576x2016xbf16>
    %5377 = stablehlo.divide %4810, %5376 : tensor<576x2016xbf16>
    %cst_971 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5378 = stablehlo.broadcast_in_dim %cst_971, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5379 = stablehlo.add %5014, %5378 : tensor<576x2016xbf16>
    %5380 = stablehlo.sqrt %5379 : tensor<576x2016xbf16>
    %cst_972 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5381 = stablehlo.broadcast_in_dim %cst_972, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5382 = stablehlo.add %5380, %5381 : tensor<576x2016xbf16>
    %5383 = stablehlo.divide %4813, %5382 : tensor<576x2016xbf16>
    %cst_973 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5384 = stablehlo.broadcast_in_dim %cst_973, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5385 = stablehlo.add %5017, %5384 : tensor<2016x576xbf16>
    %5386 = stablehlo.sqrt %5385 : tensor<2016x576xbf16>
    %cst_974 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5387 = stablehlo.broadcast_in_dim %cst_974, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5388 = stablehlo.add %5386, %5387 : tensor<2016x576xbf16>
    %5389 = stablehlo.divide %4816, %5388 : tensor<2016x576xbf16>
    %cst_975 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5390 = stablehlo.broadcast_in_dim %cst_975, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5391 = stablehlo.add %5020, %5390 : tensor<576xbf16>
    %5392 = stablehlo.sqrt %5391 : tensor<576xbf16>
    %cst_976 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5393 = stablehlo.broadcast_in_dim %cst_976, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5394 = stablehlo.add %5392, %5393 : tensor<576xbf16>
    %5395 = stablehlo.divide %4819, %5394 : tensor<576xbf16>
    %cst_977 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5396 = stablehlo.broadcast_in_dim %cst_977, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5397 = stablehlo.add %5023, %5396 : tensor<576xbf16>
    %5398 = stablehlo.sqrt %5397 : tensor<576xbf16>
    %cst_978 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5399 = stablehlo.broadcast_in_dim %cst_978, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5400 = stablehlo.add %5398, %5399 : tensor<576xbf16>
    %5401 = stablehlo.divide %4822, %5400 : tensor<576xbf16>
    %cst_979 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5402 = stablehlo.broadcast_in_dim %cst_979, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5403 = stablehlo.add %5026, %5402 : tensor<576x9x128xbf16>
    %5404 = stablehlo.sqrt %5403 : tensor<576x9x128xbf16>
    %cst_980 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5405 = stablehlo.broadcast_in_dim %cst_980, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5406 = stablehlo.add %5404, %5405 : tensor<576x9x128xbf16>
    %5407 = stablehlo.divide %4825, %5406 : tensor<576x9x128xbf16>
    %cst_981 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5408 = stablehlo.broadcast_in_dim %cst_981, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5409 = stablehlo.add %5029, %5408 : tensor<9x128x576xbf16>
    %5410 = stablehlo.sqrt %5409 : tensor<9x128x576xbf16>
    %cst_982 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5411 = stablehlo.broadcast_in_dim %cst_982, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5412 = stablehlo.add %5410, %5411 : tensor<9x128x576xbf16>
    %5413 = stablehlo.divide %4828, %5412 : tensor<9x128x576xbf16>
    %cst_983 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5414 = stablehlo.broadcast_in_dim %cst_983, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5415 = stablehlo.add %5032, %5414 : tensor<576x9x128xbf16>
    %5416 = stablehlo.sqrt %5415 : tensor<576x9x128xbf16>
    %cst_984 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5417 = stablehlo.broadcast_in_dim %cst_984, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5418 = stablehlo.add %5416, %5417 : tensor<576x9x128xbf16>
    %5419 = stablehlo.divide %4831, %5418 : tensor<576x9x128xbf16>
    %cst_985 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5420 = stablehlo.broadcast_in_dim %cst_985, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5421 = stablehlo.add %5035, %5420 : tensor<576x9x128xbf16>
    %5422 = stablehlo.sqrt %5421 : tensor<576x9x128xbf16>
    %cst_986 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5423 = stablehlo.broadcast_in_dim %cst_986, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5424 = stablehlo.add %5422, %5423 : tensor<576x9x128xbf16>
    %5425 = stablehlo.divide %4834, %5424 : tensor<576x9x128xbf16>
    %cst_987 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5426 = stablehlo.broadcast_in_dim %cst_987, dims = [] : (tensor<bf16>) -> tensor<576x128256xbf16>
    %5427 = stablehlo.add %5038, %5426 : tensor<576x128256xbf16>
    %5428 = stablehlo.sqrt %5427 : tensor<576x128256xbf16>
    %cst_988 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5429 = stablehlo.broadcast_in_dim %cst_988, dims = [] : (tensor<bf16>) -> tensor<576x128256xbf16>
    %5430 = stablehlo.add %5428, %5429 : tensor<576x128256xbf16>
    %5431 = stablehlo.divide %4837, %5430 : tensor<576x128256xbf16>
    %cst_989 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5432 = stablehlo.broadcast_in_dim %cst_989, dims = [] : (tensor<bf16>) -> tensor<128256x576xbf16>
    %5433 = stablehlo.add %5041, %5432 : tensor<128256x576xbf16>
    %5434 = stablehlo.sqrt %5433 : tensor<128256x576xbf16>
    %cst_990 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %5435 = stablehlo.broadcast_in_dim %cst_990, dims = [] : (tensor<bf16>) -> tensor<128256x576xbf16>
    %5436 = stablehlo.add %5434, %5435 : tensor<128256x576xbf16>
    %5437 = stablehlo.divide %4840, %5436 : tensor<128256x576xbf16>
    %cst_991 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5438 = stablehlo.broadcast_in_dim %cst_991, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5439 = stablehlo.multiply %5438, %arg1 : tensor<576xbf16>
    %5440 = stablehlo.add %5047, %5439 : tensor<576xbf16>
    %cst_992 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5441 = stablehlo.broadcast_in_dim %cst_992, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5442 = stablehlo.multiply %5441, %arg2 : tensor<576x2016xbf16>
    %5443 = stablehlo.add %5053, %5442 : tensor<576x2016xbf16>
    %cst_993 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5444 = stablehlo.broadcast_in_dim %cst_993, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5445 = stablehlo.multiply %5444, %arg3 : tensor<576x2016xbf16>
    %5446 = stablehlo.add %5059, %5445 : tensor<576x2016xbf16>
    %cst_994 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5447 = stablehlo.broadcast_in_dim %cst_994, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5448 = stablehlo.multiply %5447, %arg4 : tensor<2016x576xbf16>
    %5449 = stablehlo.add %5065, %5448 : tensor<2016x576xbf16>
    %cst_995 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5450 = stablehlo.broadcast_in_dim %cst_995, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5451 = stablehlo.multiply %5450, %arg5 : tensor<576xbf16>
    %5452 = stablehlo.add %5071, %5451 : tensor<576xbf16>
    %cst_996 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5453 = stablehlo.broadcast_in_dim %cst_996, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5454 = stablehlo.multiply %5453, %arg6 : tensor<576xbf16>
    %5455 = stablehlo.add %5077, %5454 : tensor<576xbf16>
    %cst_997 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5456 = stablehlo.broadcast_in_dim %cst_997, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5457 = stablehlo.multiply %5456, %arg7 : tensor<576x9x128xbf16>
    %5458 = stablehlo.add %5083, %5457 : tensor<576x9x128xbf16>
    %cst_998 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5459 = stablehlo.broadcast_in_dim %cst_998, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5460 = stablehlo.multiply %5459, %arg8 : tensor<9x128x576xbf16>
    %5461 = stablehlo.add %5089, %5460 : tensor<9x128x576xbf16>
    %cst_999 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5462 = stablehlo.broadcast_in_dim %cst_999, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5463 = stablehlo.multiply %5462, %arg9 : tensor<576x9x128xbf16>
    %5464 = stablehlo.add %5095, %5463 : tensor<576x9x128xbf16>
    %cst_1000 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5465 = stablehlo.broadcast_in_dim %cst_1000, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5466 = stablehlo.multiply %5465, %arg10 : tensor<576x9x128xbf16>
    %5467 = stablehlo.add %5101, %5466 : tensor<576x9x128xbf16>
    %cst_1001 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5468 = stablehlo.broadcast_in_dim %cst_1001, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5469 = stablehlo.multiply %5468, %arg11 : tensor<576x2016xbf16>
    %5470 = stablehlo.add %5107, %5469 : tensor<576x2016xbf16>
    %cst_1002 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5471 = stablehlo.broadcast_in_dim %cst_1002, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5472 = stablehlo.multiply %5471, %arg12 : tensor<576x2016xbf16>
    %5473 = stablehlo.add %5113, %5472 : tensor<576x2016xbf16>
    %cst_1003 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5474 = stablehlo.broadcast_in_dim %cst_1003, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5475 = stablehlo.multiply %5474, %arg13 : tensor<2016x576xbf16>
    %5476 = stablehlo.add %5119, %5475 : tensor<2016x576xbf16>
    %cst_1004 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5477 = stablehlo.broadcast_in_dim %cst_1004, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5478 = stablehlo.multiply %5477, %arg14 : tensor<576xbf16>
    %5479 = stablehlo.add %5125, %5478 : tensor<576xbf16>
    %cst_1005 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5480 = stablehlo.broadcast_in_dim %cst_1005, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5481 = stablehlo.multiply %5480, %arg15 : tensor<576xbf16>
    %5482 = stablehlo.add %5131, %5481 : tensor<576xbf16>
    %cst_1006 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5483 = stablehlo.broadcast_in_dim %cst_1006, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5484 = stablehlo.multiply %5483, %arg16 : tensor<576x9x128xbf16>
    %5485 = stablehlo.add %5137, %5484 : tensor<576x9x128xbf16>
    %cst_1007 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5486 = stablehlo.broadcast_in_dim %cst_1007, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5487 = stablehlo.multiply %5486, %arg17 : tensor<9x128x576xbf16>
    %5488 = stablehlo.add %5143, %5487 : tensor<9x128x576xbf16>
    %cst_1008 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5489 = stablehlo.broadcast_in_dim %cst_1008, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5490 = stablehlo.multiply %5489, %arg18 : tensor<576x9x128xbf16>
    %5491 = stablehlo.add %5149, %5490 : tensor<576x9x128xbf16>
    %cst_1009 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5492 = stablehlo.broadcast_in_dim %cst_1009, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5493 = stablehlo.multiply %5492, %arg19 : tensor<576x9x128xbf16>
    %5494 = stablehlo.add %5155, %5493 : tensor<576x9x128xbf16>
    %cst_1010 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5495 = stablehlo.broadcast_in_dim %cst_1010, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5496 = stablehlo.multiply %5495, %arg20 : tensor<576x2016xbf16>
    %5497 = stablehlo.add %5161, %5496 : tensor<576x2016xbf16>
    %cst_1011 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5498 = stablehlo.broadcast_in_dim %cst_1011, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5499 = stablehlo.multiply %5498, %arg21 : tensor<576x2016xbf16>
    %5500 = stablehlo.add %5167, %5499 : tensor<576x2016xbf16>
    %cst_1012 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5501 = stablehlo.broadcast_in_dim %cst_1012, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5502 = stablehlo.multiply %5501, %arg22 : tensor<2016x576xbf16>
    %5503 = stablehlo.add %5173, %5502 : tensor<2016x576xbf16>
    %cst_1013 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5504 = stablehlo.broadcast_in_dim %cst_1013, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5505 = stablehlo.multiply %5504, %arg23 : tensor<576xbf16>
    %5506 = stablehlo.add %5179, %5505 : tensor<576xbf16>
    %cst_1014 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5507 = stablehlo.broadcast_in_dim %cst_1014, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5508 = stablehlo.multiply %5507, %arg24 : tensor<576xbf16>
    %5509 = stablehlo.add %5185, %5508 : tensor<576xbf16>
    %cst_1015 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5510 = stablehlo.broadcast_in_dim %cst_1015, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5511 = stablehlo.multiply %5510, %arg25 : tensor<576x9x128xbf16>
    %5512 = stablehlo.add %5191, %5511 : tensor<576x9x128xbf16>
    %cst_1016 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5513 = stablehlo.broadcast_in_dim %cst_1016, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5514 = stablehlo.multiply %5513, %arg26 : tensor<9x128x576xbf16>
    %5515 = stablehlo.add %5197, %5514 : tensor<9x128x576xbf16>
    %cst_1017 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5516 = stablehlo.broadcast_in_dim %cst_1017, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5517 = stablehlo.multiply %5516, %arg27 : tensor<576x9x128xbf16>
    %5518 = stablehlo.add %5203, %5517 : tensor<576x9x128xbf16>
    %cst_1018 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5519 = stablehlo.broadcast_in_dim %cst_1018, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5520 = stablehlo.multiply %5519, %arg28 : tensor<576x9x128xbf16>
    %5521 = stablehlo.add %5209, %5520 : tensor<576x9x128xbf16>
    %cst_1019 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5522 = stablehlo.broadcast_in_dim %cst_1019, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5523 = stablehlo.multiply %5522, %arg29 : tensor<576x2016xbf16>
    %5524 = stablehlo.add %5215, %5523 : tensor<576x2016xbf16>
    %cst_1020 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5525 = stablehlo.broadcast_in_dim %cst_1020, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5526 = stablehlo.multiply %5525, %arg30 : tensor<576x2016xbf16>
    %5527 = stablehlo.add %5221, %5526 : tensor<576x2016xbf16>
    %cst_1021 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5528 = stablehlo.broadcast_in_dim %cst_1021, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5529 = stablehlo.multiply %5528, %arg31 : tensor<2016x576xbf16>
    %5530 = stablehlo.add %5227, %5529 : tensor<2016x576xbf16>
    %cst_1022 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5531 = stablehlo.broadcast_in_dim %cst_1022, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5532 = stablehlo.multiply %5531, %arg32 : tensor<576xbf16>
    %5533 = stablehlo.add %5233, %5532 : tensor<576xbf16>
    %cst_1023 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5534 = stablehlo.broadcast_in_dim %cst_1023, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5535 = stablehlo.multiply %5534, %arg33 : tensor<576xbf16>
    %5536 = stablehlo.add %5239, %5535 : tensor<576xbf16>
    %cst_1024 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5537 = stablehlo.broadcast_in_dim %cst_1024, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5538 = stablehlo.multiply %5537, %arg34 : tensor<576x9x128xbf16>
    %5539 = stablehlo.add %5245, %5538 : tensor<576x9x128xbf16>
    %cst_1025 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5540 = stablehlo.broadcast_in_dim %cst_1025, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5541 = stablehlo.multiply %5540, %arg35 : tensor<9x128x576xbf16>
    %5542 = stablehlo.add %5251, %5541 : tensor<9x128x576xbf16>
    %cst_1026 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5543 = stablehlo.broadcast_in_dim %cst_1026, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5544 = stablehlo.multiply %5543, %arg36 : tensor<576x9x128xbf16>
    %5545 = stablehlo.add %5257, %5544 : tensor<576x9x128xbf16>
    %cst_1027 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5546 = stablehlo.broadcast_in_dim %cst_1027, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5547 = stablehlo.multiply %5546, %arg37 : tensor<576x9x128xbf16>
    %5548 = stablehlo.add %5263, %5547 : tensor<576x9x128xbf16>
    %cst_1028 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5549 = stablehlo.broadcast_in_dim %cst_1028, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5550 = stablehlo.multiply %5549, %arg38 : tensor<576x2016xbf16>
    %5551 = stablehlo.add %5269, %5550 : tensor<576x2016xbf16>
    %cst_1029 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5552 = stablehlo.broadcast_in_dim %cst_1029, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5553 = stablehlo.multiply %5552, %arg39 : tensor<576x2016xbf16>
    %5554 = stablehlo.add %5275, %5553 : tensor<576x2016xbf16>
    %cst_1030 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5555 = stablehlo.broadcast_in_dim %cst_1030, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5556 = stablehlo.multiply %5555, %arg40 : tensor<2016x576xbf16>
    %5557 = stablehlo.add %5281, %5556 : tensor<2016x576xbf16>
    %cst_1031 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5558 = stablehlo.broadcast_in_dim %cst_1031, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5559 = stablehlo.multiply %5558, %arg41 : tensor<576xbf16>
    %5560 = stablehlo.add %5287, %5559 : tensor<576xbf16>
    %cst_1032 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5561 = stablehlo.broadcast_in_dim %cst_1032, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5562 = stablehlo.multiply %5561, %arg42 : tensor<576xbf16>
    %5563 = stablehlo.add %5293, %5562 : tensor<576xbf16>
    %cst_1033 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5564 = stablehlo.broadcast_in_dim %cst_1033, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5565 = stablehlo.multiply %5564, %arg43 : tensor<576x9x128xbf16>
    %5566 = stablehlo.add %5299, %5565 : tensor<576x9x128xbf16>
    %cst_1034 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5567 = stablehlo.broadcast_in_dim %cst_1034, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5568 = stablehlo.multiply %5567, %arg44 : tensor<9x128x576xbf16>
    %5569 = stablehlo.add %5305, %5568 : tensor<9x128x576xbf16>
    %cst_1035 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5570 = stablehlo.broadcast_in_dim %cst_1035, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5571 = stablehlo.multiply %5570, %arg45 : tensor<576x9x128xbf16>
    %5572 = stablehlo.add %5311, %5571 : tensor<576x9x128xbf16>
    %cst_1036 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5573 = stablehlo.broadcast_in_dim %cst_1036, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5574 = stablehlo.multiply %5573, %arg46 : tensor<576x9x128xbf16>
    %5575 = stablehlo.add %5317, %5574 : tensor<576x9x128xbf16>
    %cst_1037 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5576 = stablehlo.broadcast_in_dim %cst_1037, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5577 = stablehlo.multiply %5576, %arg47 : tensor<576x2016xbf16>
    %5578 = stablehlo.add %5323, %5577 : tensor<576x2016xbf16>
    %cst_1038 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5579 = stablehlo.broadcast_in_dim %cst_1038, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5580 = stablehlo.multiply %5579, %arg48 : tensor<576x2016xbf16>
    %5581 = stablehlo.add %5329, %5580 : tensor<576x2016xbf16>
    %cst_1039 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5582 = stablehlo.broadcast_in_dim %cst_1039, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5583 = stablehlo.multiply %5582, %arg49 : tensor<2016x576xbf16>
    %5584 = stablehlo.add %5335, %5583 : tensor<2016x576xbf16>
    %cst_1040 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5585 = stablehlo.broadcast_in_dim %cst_1040, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5586 = stablehlo.multiply %5585, %arg50 : tensor<576xbf16>
    %5587 = stablehlo.add %5341, %5586 : tensor<576xbf16>
    %cst_1041 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5588 = stablehlo.broadcast_in_dim %cst_1041, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5589 = stablehlo.multiply %5588, %arg51 : tensor<576xbf16>
    %5590 = stablehlo.add %5347, %5589 : tensor<576xbf16>
    %cst_1042 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5591 = stablehlo.broadcast_in_dim %cst_1042, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5592 = stablehlo.multiply %5591, %arg52 : tensor<576x9x128xbf16>
    %5593 = stablehlo.add %5353, %5592 : tensor<576x9x128xbf16>
    %cst_1043 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5594 = stablehlo.broadcast_in_dim %cst_1043, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5595 = stablehlo.multiply %5594, %arg53 : tensor<9x128x576xbf16>
    %5596 = stablehlo.add %5359, %5595 : tensor<9x128x576xbf16>
    %cst_1044 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5597 = stablehlo.broadcast_in_dim %cst_1044, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5598 = stablehlo.multiply %5597, %arg54 : tensor<576x9x128xbf16>
    %5599 = stablehlo.add %5365, %5598 : tensor<576x9x128xbf16>
    %cst_1045 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5600 = stablehlo.broadcast_in_dim %cst_1045, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5601 = stablehlo.multiply %5600, %arg55 : tensor<576x9x128xbf16>
    %5602 = stablehlo.add %5371, %5601 : tensor<576x9x128xbf16>
    %cst_1046 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5603 = stablehlo.broadcast_in_dim %cst_1046, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5604 = stablehlo.multiply %5603, %arg56 : tensor<576x2016xbf16>
    %5605 = stablehlo.add %5377, %5604 : tensor<576x2016xbf16>
    %cst_1047 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5606 = stablehlo.broadcast_in_dim %cst_1047, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5607 = stablehlo.multiply %5606, %arg57 : tensor<576x2016xbf16>
    %5608 = stablehlo.add %5383, %5607 : tensor<576x2016xbf16>
    %cst_1048 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5609 = stablehlo.broadcast_in_dim %cst_1048, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5610 = stablehlo.multiply %5609, %arg58 : tensor<2016x576xbf16>
    %5611 = stablehlo.add %5389, %5610 : tensor<2016x576xbf16>
    %cst_1049 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5612 = stablehlo.broadcast_in_dim %cst_1049, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5613 = stablehlo.multiply %5612, %arg59 : tensor<576xbf16>
    %5614 = stablehlo.add %5395, %5613 : tensor<576xbf16>
    %cst_1050 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5615 = stablehlo.broadcast_in_dim %cst_1050, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5616 = stablehlo.multiply %5615, %arg60 : tensor<576xbf16>
    %5617 = stablehlo.add %5401, %5616 : tensor<576xbf16>
    %cst_1051 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5618 = stablehlo.broadcast_in_dim %cst_1051, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5619 = stablehlo.multiply %5618, %arg61 : tensor<576x9x128xbf16>
    %5620 = stablehlo.add %5407, %5619 : tensor<576x9x128xbf16>
    %cst_1052 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5621 = stablehlo.broadcast_in_dim %cst_1052, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5622 = stablehlo.multiply %5621, %arg62 : tensor<9x128x576xbf16>
    %5623 = stablehlo.add %5413, %5622 : tensor<9x128x576xbf16>
    %cst_1053 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5624 = stablehlo.broadcast_in_dim %cst_1053, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5625 = stablehlo.multiply %5624, %arg63 : tensor<576x9x128xbf16>
    %5626 = stablehlo.add %5419, %5625 : tensor<576x9x128xbf16>
    %cst_1054 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5627 = stablehlo.broadcast_in_dim %cst_1054, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5628 = stablehlo.multiply %5627, %arg64 : tensor<576x9x128xbf16>
    %5629 = stablehlo.add %5425, %5628 : tensor<576x9x128xbf16>
    %cst_1055 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5630 = stablehlo.broadcast_in_dim %cst_1055, dims = [] : (tensor<bf16>) -> tensor<576x128256xbf16>
    %5631 = stablehlo.multiply %5630, %arg65 : tensor<576x128256xbf16>
    %5632 = stablehlo.add %5431, %5631 : tensor<576x128256xbf16>
    %cst_1056 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %5633 = stablehlo.broadcast_in_dim %cst_1056, dims = [] : (tensor<bf16>) -> tensor<128256x576xbf16>
    %5634 = stablehlo.multiply %5633, %arg66 : tensor<128256x576xbf16>
    %5635 = stablehlo.add %5437, %5634 : tensor<128256x576xbf16>
    %c_1057 = stablehlo.constant dense<0> : tensor<i32>
    %5636 = stablehlo.subtract %arg200, %c_1057 : tensor<i32>
    %c_1058 = stablehlo.constant dense<0> : tensor<i32>
    %c_1059 = stablehlo.constant dense<3> : tensor<i32>
    %5637 = call @clip(%5636, %c_1058, %c_1059) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5638 = stablehlo.convert %5637 : (tensor<i32>) -> tensor<f32>
    %cst_1060 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %5639 = stablehlo.divide %5638, %cst_1060 : tensor<f32>
    %cst_1061 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5640 = stablehlo.subtract %cst_1061, %5639 : tensor<f32>
    %cst_1062 = stablehlo.constant dense<-3.000000e-05> : tensor<f32>
    %5641 = stablehlo.multiply %cst_1062, %5640 : tensor<f32>
    %cst_1063 = stablehlo.constant dense<3.000000e-05> : tensor<f32>
    %5642 = stablehlo.add %5641, %cst_1063 : tensor<f32>
    %c_1064 = stablehlo.constant dense<3> : tensor<i32>
    %5643 = stablehlo.compare  LT, %arg200, %c_1064,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_1065 = stablehlo.constant dense<3> : tensor<i32>
    %5644 = stablehlo.subtract %arg200, %c_1065 : tensor<i32>
    %5645 = stablehlo.convert %5644 : (tensor<i32>) -> tensor<f32>
    %cst_1066 = stablehlo.constant dense<2.700000e+01> : tensor<f32>
    %5646 = stablehlo.divide %5645, %cst_1066 : tensor<f32>
    %cst_1067 = stablehlo.constant dense<3.14159274> : tensor<f32>
    %5647 = stablehlo.multiply %cst_1067, %5646 : tensor<f32>
    %5648 = stablehlo.cosine %5647 : tensor<f32>
    %cst_1068 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5649 = stablehlo.add %5648, %cst_1068 : tensor<f32>
    %cst_1069 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %5650 = stablehlo.multiply %cst_1069, %5649 : tensor<f32>
    %cst_1070 = stablehlo.constant dense<3.000000e-05> : tensor<f32>
    %5651 = stablehlo.multiply %cst_1070, %5650 : tensor<f32>
    %cst_1071 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5652 = stablehlo.subtract %cst_1071, %5650 : tensor<f32>
    %cst_1072 = stablehlo.constant dense<3.000000e-06> : tensor<f32>
    %5653 = stablehlo.multiply %cst_1072, %5652 : tensor<f32>
    %5654 = stablehlo.add %5651, %5653 : tensor<f32>
    %5655 = call @_where_293(%5643, %5642, %5654) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %cst_1073 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %5656 = stablehlo.multiply %cst_1073, %5655 : tensor<f32>
    %5657 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5658 = stablehlo.broadcast_in_dim %5657, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5659 = stablehlo.multiply %5658, %5440 : tensor<576xbf16>
    %5660 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5661 = stablehlo.broadcast_in_dim %5660, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5662 = stablehlo.multiply %5661, %5443 : tensor<576x2016xbf16>
    %5663 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5664 = stablehlo.broadcast_in_dim %5663, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5665 = stablehlo.multiply %5664, %5446 : tensor<576x2016xbf16>
    %5666 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5667 = stablehlo.broadcast_in_dim %5666, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5668 = stablehlo.multiply %5667, %5449 : tensor<2016x576xbf16>
    %5669 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5670 = stablehlo.broadcast_in_dim %5669, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5671 = stablehlo.multiply %5670, %5452 : tensor<576xbf16>
    %5672 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5673 = stablehlo.broadcast_in_dim %5672, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5674 = stablehlo.multiply %5673, %5455 : tensor<576xbf16>
    %5675 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5676 = stablehlo.broadcast_in_dim %5675, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5677 = stablehlo.multiply %5676, %5458 : tensor<576x9x128xbf16>
    %5678 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5679 = stablehlo.broadcast_in_dim %5678, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5680 = stablehlo.multiply %5679, %5461 : tensor<9x128x576xbf16>
    %5681 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5682 = stablehlo.broadcast_in_dim %5681, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5683 = stablehlo.multiply %5682, %5464 : tensor<576x9x128xbf16>
    %5684 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5685 = stablehlo.broadcast_in_dim %5684, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5686 = stablehlo.multiply %5685, %5467 : tensor<576x9x128xbf16>
    %5687 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5688 = stablehlo.broadcast_in_dim %5687, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5689 = stablehlo.multiply %5688, %5470 : tensor<576x2016xbf16>
    %5690 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5691 = stablehlo.broadcast_in_dim %5690, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5692 = stablehlo.multiply %5691, %5473 : tensor<576x2016xbf16>
    %5693 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5694 = stablehlo.broadcast_in_dim %5693, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5695 = stablehlo.multiply %5694, %5476 : tensor<2016x576xbf16>
    %5696 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5697 = stablehlo.broadcast_in_dim %5696, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5698 = stablehlo.multiply %5697, %5479 : tensor<576xbf16>
    %5699 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5700 = stablehlo.broadcast_in_dim %5699, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5701 = stablehlo.multiply %5700, %5482 : tensor<576xbf16>
    %5702 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5703 = stablehlo.broadcast_in_dim %5702, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5704 = stablehlo.multiply %5703, %5485 : tensor<576x9x128xbf16>
    %5705 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5706 = stablehlo.broadcast_in_dim %5705, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5707 = stablehlo.multiply %5706, %5488 : tensor<9x128x576xbf16>
    %5708 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5709 = stablehlo.broadcast_in_dim %5708, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5710 = stablehlo.multiply %5709, %5491 : tensor<576x9x128xbf16>
    %5711 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5712 = stablehlo.broadcast_in_dim %5711, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5713 = stablehlo.multiply %5712, %5494 : tensor<576x9x128xbf16>
    %5714 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5715 = stablehlo.broadcast_in_dim %5714, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5716 = stablehlo.multiply %5715, %5497 : tensor<576x2016xbf16>
    %5717 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5718 = stablehlo.broadcast_in_dim %5717, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5719 = stablehlo.multiply %5718, %5500 : tensor<576x2016xbf16>
    %5720 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5721 = stablehlo.broadcast_in_dim %5720, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5722 = stablehlo.multiply %5721, %5503 : tensor<2016x576xbf16>
    %5723 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5724 = stablehlo.broadcast_in_dim %5723, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5725 = stablehlo.multiply %5724, %5506 : tensor<576xbf16>
    %5726 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5727 = stablehlo.broadcast_in_dim %5726, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5728 = stablehlo.multiply %5727, %5509 : tensor<576xbf16>
    %5729 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5730 = stablehlo.broadcast_in_dim %5729, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5731 = stablehlo.multiply %5730, %5512 : tensor<576x9x128xbf16>
    %5732 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5733 = stablehlo.broadcast_in_dim %5732, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5734 = stablehlo.multiply %5733, %5515 : tensor<9x128x576xbf16>
    %5735 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5736 = stablehlo.broadcast_in_dim %5735, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5737 = stablehlo.multiply %5736, %5518 : tensor<576x9x128xbf16>
    %5738 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5739 = stablehlo.broadcast_in_dim %5738, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5740 = stablehlo.multiply %5739, %5521 : tensor<576x9x128xbf16>
    %5741 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5742 = stablehlo.broadcast_in_dim %5741, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5743 = stablehlo.multiply %5742, %5524 : tensor<576x2016xbf16>
    %5744 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5745 = stablehlo.broadcast_in_dim %5744, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5746 = stablehlo.multiply %5745, %5527 : tensor<576x2016xbf16>
    %5747 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5748 = stablehlo.broadcast_in_dim %5747, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5749 = stablehlo.multiply %5748, %5530 : tensor<2016x576xbf16>
    %5750 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5751 = stablehlo.broadcast_in_dim %5750, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5752 = stablehlo.multiply %5751, %5533 : tensor<576xbf16>
    %5753 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5754 = stablehlo.broadcast_in_dim %5753, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5755 = stablehlo.multiply %5754, %5536 : tensor<576xbf16>
    %5756 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5757 = stablehlo.broadcast_in_dim %5756, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5758 = stablehlo.multiply %5757, %5539 : tensor<576x9x128xbf16>
    %5759 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5760 = stablehlo.broadcast_in_dim %5759, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5761 = stablehlo.multiply %5760, %5542 : tensor<9x128x576xbf16>
    %5762 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5763 = stablehlo.broadcast_in_dim %5762, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5764 = stablehlo.multiply %5763, %5545 : tensor<576x9x128xbf16>
    %5765 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5766 = stablehlo.broadcast_in_dim %5765, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5767 = stablehlo.multiply %5766, %5548 : tensor<576x9x128xbf16>
    %5768 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5769 = stablehlo.broadcast_in_dim %5768, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5770 = stablehlo.multiply %5769, %5551 : tensor<576x2016xbf16>
    %5771 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5772 = stablehlo.broadcast_in_dim %5771, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5773 = stablehlo.multiply %5772, %5554 : tensor<576x2016xbf16>
    %5774 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5775 = stablehlo.broadcast_in_dim %5774, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5776 = stablehlo.multiply %5775, %5557 : tensor<2016x576xbf16>
    %5777 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5778 = stablehlo.broadcast_in_dim %5777, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5779 = stablehlo.multiply %5778, %5560 : tensor<576xbf16>
    %5780 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5781 = stablehlo.broadcast_in_dim %5780, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5782 = stablehlo.multiply %5781, %5563 : tensor<576xbf16>
    %5783 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5784 = stablehlo.broadcast_in_dim %5783, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5785 = stablehlo.multiply %5784, %5566 : tensor<576x9x128xbf16>
    %5786 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5787 = stablehlo.broadcast_in_dim %5786, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5788 = stablehlo.multiply %5787, %5569 : tensor<9x128x576xbf16>
    %5789 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5790 = stablehlo.broadcast_in_dim %5789, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5791 = stablehlo.multiply %5790, %5572 : tensor<576x9x128xbf16>
    %5792 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5793 = stablehlo.broadcast_in_dim %5792, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5794 = stablehlo.multiply %5793, %5575 : tensor<576x9x128xbf16>
    %5795 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5796 = stablehlo.broadcast_in_dim %5795, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5797 = stablehlo.multiply %5796, %5578 : tensor<576x2016xbf16>
    %5798 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5799 = stablehlo.broadcast_in_dim %5798, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5800 = stablehlo.multiply %5799, %5581 : tensor<576x2016xbf16>
    %5801 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5802 = stablehlo.broadcast_in_dim %5801, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5803 = stablehlo.multiply %5802, %5584 : tensor<2016x576xbf16>
    %5804 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5805 = stablehlo.broadcast_in_dim %5804, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5806 = stablehlo.multiply %5805, %5587 : tensor<576xbf16>
    %5807 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5808 = stablehlo.broadcast_in_dim %5807, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5809 = stablehlo.multiply %5808, %5590 : tensor<576xbf16>
    %5810 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5811 = stablehlo.broadcast_in_dim %5810, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5812 = stablehlo.multiply %5811, %5593 : tensor<576x9x128xbf16>
    %5813 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5814 = stablehlo.broadcast_in_dim %5813, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5815 = stablehlo.multiply %5814, %5596 : tensor<9x128x576xbf16>
    %5816 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5817 = stablehlo.broadcast_in_dim %5816, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5818 = stablehlo.multiply %5817, %5599 : tensor<576x9x128xbf16>
    %5819 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5820 = stablehlo.broadcast_in_dim %5819, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5821 = stablehlo.multiply %5820, %5602 : tensor<576x9x128xbf16>
    %5822 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5823 = stablehlo.broadcast_in_dim %5822, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5824 = stablehlo.multiply %5823, %5605 : tensor<576x2016xbf16>
    %5825 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5826 = stablehlo.broadcast_in_dim %5825, dims = [] : (tensor<bf16>) -> tensor<576x2016xbf16>
    %5827 = stablehlo.multiply %5826, %5608 : tensor<576x2016xbf16>
    %5828 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5829 = stablehlo.broadcast_in_dim %5828, dims = [] : (tensor<bf16>) -> tensor<2016x576xbf16>
    %5830 = stablehlo.multiply %5829, %5611 : tensor<2016x576xbf16>
    %5831 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5832 = stablehlo.broadcast_in_dim %5831, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5833 = stablehlo.multiply %5832, %5614 : tensor<576xbf16>
    %5834 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5835 = stablehlo.broadcast_in_dim %5834, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %5836 = stablehlo.multiply %5835, %5617 : tensor<576xbf16>
    %5837 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5838 = stablehlo.broadcast_in_dim %5837, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5839 = stablehlo.multiply %5838, %5620 : tensor<576x9x128xbf16>
    %5840 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5841 = stablehlo.broadcast_in_dim %5840, dims = [] : (tensor<bf16>) -> tensor<9x128x576xbf16>
    %5842 = stablehlo.multiply %5841, %5623 : tensor<9x128x576xbf16>
    %5843 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5844 = stablehlo.broadcast_in_dim %5843, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5845 = stablehlo.multiply %5844, %5626 : tensor<576x9x128xbf16>
    %5846 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5847 = stablehlo.broadcast_in_dim %5846, dims = [] : (tensor<bf16>) -> tensor<576x9x128xbf16>
    %5848 = stablehlo.multiply %5847, %5629 : tensor<576x9x128xbf16>
    %5849 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5850 = stablehlo.broadcast_in_dim %5849, dims = [] : (tensor<bf16>) -> tensor<576x128256xbf16>
    %5851 = stablehlo.multiply %5850, %5632 : tensor<576x128256xbf16>
    %5852 = stablehlo.convert %5656 : (tensor<f32>) -> tensor<bf16>
    %5853 = stablehlo.broadcast_in_dim %5852, dims = [] : (tensor<bf16>) -> tensor<128256x576xbf16>
    %5854 = stablehlo.multiply %5853, %5635 : tensor<128256x576xbf16>
    %c_1074 = stablehlo.constant dense<2147483647> : tensor<i32>
    %5855 = stablehlo.compare  LT, %arg200, %c_1074,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_1075 = stablehlo.constant dense<1> : tensor<i32>
    %5856 = stablehlo.add %arg200, %c_1075 : tensor<i32>
    %c_1076 = stablehlo.constant dense<2147483647> : tensor<i32>
    %5857 = call @_where_260(%5855, %5856, %c_1076) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5858 = stablehlo.add %arg1, %5659 : tensor<576xbf16>
    %5859 = stablehlo.add %arg2, %5662 : tensor<576x2016xbf16>
    %5860 = stablehlo.add %arg3, %5665 : tensor<576x2016xbf16>
    %5861 = stablehlo.add %arg4, %5668 : tensor<2016x576xbf16>
    %5862 = stablehlo.add %arg5, %5671 : tensor<576xbf16>
    %5863 = stablehlo.add %arg6, %5674 : tensor<576xbf16>
    %5864 = stablehlo.add %arg7, %5677 : tensor<576x9x128xbf16>
    %5865 = stablehlo.add %arg8, %5680 : tensor<9x128x576xbf16>
    %5866 = stablehlo.add %arg9, %5683 : tensor<576x9x128xbf16>
    %5867 = stablehlo.add %arg10, %5686 : tensor<576x9x128xbf16>
    %5868 = stablehlo.add %arg11, %5689 : tensor<576x2016xbf16>
    %5869 = stablehlo.add %arg12, %5692 : tensor<576x2016xbf16>
    %5870 = stablehlo.add %arg13, %5695 : tensor<2016x576xbf16>
    %5871 = stablehlo.add %arg14, %5698 : tensor<576xbf16>
    %5872 = stablehlo.add %arg15, %5701 : tensor<576xbf16>
    %5873 = stablehlo.add %arg16, %5704 : tensor<576x9x128xbf16>
    %5874 = stablehlo.add %arg17, %5707 : tensor<9x128x576xbf16>
    %5875 = stablehlo.add %arg18, %5710 : tensor<576x9x128xbf16>
    %5876 = stablehlo.add %arg19, %5713 : tensor<576x9x128xbf16>
    %5877 = stablehlo.add %arg20, %5716 : tensor<576x2016xbf16>
    %5878 = stablehlo.add %arg21, %5719 : tensor<576x2016xbf16>
    %5879 = stablehlo.add %arg22, %5722 : tensor<2016x576xbf16>
    %5880 = stablehlo.add %arg23, %5725 : tensor<576xbf16>
    %5881 = stablehlo.add %arg24, %5728 : tensor<576xbf16>
    %5882 = stablehlo.add %arg25, %5731 : tensor<576x9x128xbf16>
    %5883 = stablehlo.add %arg26, %5734 : tensor<9x128x576xbf16>
    %5884 = stablehlo.add %arg27, %5737 : tensor<576x9x128xbf16>
    %5885 = stablehlo.add %arg28, %5740 : tensor<576x9x128xbf16>
    %5886 = stablehlo.add %arg29, %5743 : tensor<576x2016xbf16>
    %5887 = stablehlo.add %arg30, %5746 : tensor<576x2016xbf16>
    %5888 = stablehlo.add %arg31, %5749 : tensor<2016x576xbf16>
    %5889 = stablehlo.add %arg32, %5752 : tensor<576xbf16>
    %5890 = stablehlo.add %arg33, %5755 : tensor<576xbf16>
    %5891 = stablehlo.add %arg34, %5758 : tensor<576x9x128xbf16>
    %5892 = stablehlo.add %arg35, %5761 : tensor<9x128x576xbf16>
    %5893 = stablehlo.add %arg36, %5764 : tensor<576x9x128xbf16>
    %5894 = stablehlo.add %arg37, %5767 : tensor<576x9x128xbf16>
    %5895 = stablehlo.add %arg38, %5770 : tensor<576x2016xbf16>
    %5896 = stablehlo.add %arg39, %5773 : tensor<576x2016xbf16>
    %5897 = stablehlo.add %arg40, %5776 : tensor<2016x576xbf16>
    %5898 = stablehlo.add %arg41, %5779 : tensor<576xbf16>
    %5899 = stablehlo.add %arg42, %5782 : tensor<576xbf16>
    %5900 = stablehlo.add %arg43, %5785 : tensor<576x9x128xbf16>
    %5901 = stablehlo.add %arg44, %5788 : tensor<9x128x576xbf16>
    %5902 = stablehlo.add %arg45, %5791 : tensor<576x9x128xbf16>
    %5903 = stablehlo.add %arg46, %5794 : tensor<576x9x128xbf16>
    %5904 = stablehlo.add %arg47, %5797 : tensor<576x2016xbf16>
    %5905 = stablehlo.add %arg48, %5800 : tensor<576x2016xbf16>
    %5906 = stablehlo.add %arg49, %5803 : tensor<2016x576xbf16>
    %5907 = stablehlo.add %arg50, %5806 : tensor<576xbf16>
    %5908 = stablehlo.add %arg51, %5809 : tensor<576xbf16>
    %5909 = stablehlo.add %arg52, %5812 : tensor<576x9x128xbf16>
    %5910 = stablehlo.add %arg53, %5815 : tensor<9x128x576xbf16>
    %5911 = stablehlo.add %arg54, %5818 : tensor<576x9x128xbf16>
    %5912 = stablehlo.add %arg55, %5821 : tensor<576x9x128xbf16>
    %5913 = stablehlo.add %arg56, %5824 : tensor<576x2016xbf16>
    %5914 = stablehlo.add %arg57, %5827 : tensor<576x2016xbf16>
    %5915 = stablehlo.add %arg58, %5830 : tensor<2016x576xbf16>
    %5916 = stablehlo.add %arg59, %5833 : tensor<576xbf16>
    %5917 = stablehlo.add %arg60, %5836 : tensor<576xbf16>
    %5918 = stablehlo.add %arg61, %5839 : tensor<576x9x128xbf16>
    %5919 = stablehlo.add %arg62, %5842 : tensor<9x128x576xbf16>
    %5920 = stablehlo.add %arg63, %5845 : tensor<576x9x128xbf16>
    %5921 = stablehlo.add %arg64, %5848 : tensor<576x9x128xbf16>
    %5922 = stablehlo.add %arg65, %5851 : tensor<576x128256xbf16>
    %5923 = stablehlo.add %arg66, %5854 : tensor<128256x576xbf16>
    %c_1077 = stablehlo.constant dense<1> : tensor<i32>
    %5924 = stablehlo.add %arg0, %c_1077 : tensor<i32>
    %5925 = stablehlo.multiply %3585, %3585 : tensor<576xbf16>
    %5926 = stablehlo.convert %5925 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1078 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5927 = stablehlo.reduce(%5926 init: %cst_1078) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %5928 = stablehlo.convert %5927 : (tensor<f32>) -> tensor<bf16>
    %cst_1079 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %5929 = stablehlo.add %cst_1079, %5928 : tensor<bf16>
    %5930 = stablehlo.multiply %3590, %3590 : tensor<576x2016xbf16>
    %5931 = stablehlo.convert %5930 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1080 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5932 = stablehlo.reduce(%5931 init: %cst_1080) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %5933 = stablehlo.convert %5932 : (tensor<f32>) -> tensor<bf16>
    %5934 = stablehlo.add %5929, %5933 : tensor<bf16>
    %5935 = stablehlo.multiply %3595, %3595 : tensor<576x2016xbf16>
    %5936 = stablehlo.convert %5935 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1081 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5937 = stablehlo.reduce(%5936 init: %cst_1081) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %5938 = stablehlo.convert %5937 : (tensor<f32>) -> tensor<bf16>
    %5939 = stablehlo.add %5934, %5938 : tensor<bf16>
    %5940 = stablehlo.multiply %3600, %3600 : tensor<2016x576xbf16>
    %5941 = stablehlo.convert %5940 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1082 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5942 = stablehlo.reduce(%5941 init: %cst_1082) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %5943 = stablehlo.convert %5942 : (tensor<f32>) -> tensor<bf16>
    %5944 = stablehlo.add %5939, %5943 : tensor<bf16>
    %5945 = stablehlo.multiply %3605, %3605 : tensor<576xbf16>
    %5946 = stablehlo.convert %5945 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1083 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5947 = stablehlo.reduce(%5946 init: %cst_1083) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %5948 = stablehlo.convert %5947 : (tensor<f32>) -> tensor<bf16>
    %5949 = stablehlo.add %5944, %5948 : tensor<bf16>
    %5950 = stablehlo.multiply %3610, %3610 : tensor<576xbf16>
    %5951 = stablehlo.convert %5950 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1084 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5952 = stablehlo.reduce(%5951 init: %cst_1084) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %5953 = stablehlo.convert %5952 : (tensor<f32>) -> tensor<bf16>
    %5954 = stablehlo.add %5949, %5953 : tensor<bf16>
    %5955 = stablehlo.multiply %3615, %3615 : tensor<576x9x128xbf16>
    %5956 = stablehlo.convert %5955 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1085 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5957 = stablehlo.reduce(%5956 init: %cst_1085) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %5958 = stablehlo.convert %5957 : (tensor<f32>) -> tensor<bf16>
    %5959 = stablehlo.add %5954, %5958 : tensor<bf16>
    %5960 = stablehlo.multiply %3620, %3620 : tensor<9x128x576xbf16>
    %5961 = stablehlo.convert %5960 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1086 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5962 = stablehlo.reduce(%5961 init: %cst_1086) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %5963 = stablehlo.convert %5962 : (tensor<f32>) -> tensor<bf16>
    %5964 = stablehlo.add %5959, %5963 : tensor<bf16>
    %5965 = stablehlo.multiply %3625, %3625 : tensor<576x9x128xbf16>
    %5966 = stablehlo.convert %5965 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1087 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5967 = stablehlo.reduce(%5966 init: %cst_1087) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %5968 = stablehlo.convert %5967 : (tensor<f32>) -> tensor<bf16>
    %5969 = stablehlo.add %5964, %5968 : tensor<bf16>
    %5970 = stablehlo.multiply %3630, %3630 : tensor<576x9x128xbf16>
    %5971 = stablehlo.convert %5970 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1088 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5972 = stablehlo.reduce(%5971 init: %cst_1088) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %5973 = stablehlo.convert %5972 : (tensor<f32>) -> tensor<bf16>
    %5974 = stablehlo.add %5969, %5973 : tensor<bf16>
    %5975 = stablehlo.multiply %3635, %3635 : tensor<576x2016xbf16>
    %5976 = stablehlo.convert %5975 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1089 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5977 = stablehlo.reduce(%5976 init: %cst_1089) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %5978 = stablehlo.convert %5977 : (tensor<f32>) -> tensor<bf16>
    %5979 = stablehlo.add %5974, %5978 : tensor<bf16>
    %5980 = stablehlo.multiply %3640, %3640 : tensor<576x2016xbf16>
    %5981 = stablehlo.convert %5980 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1090 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5982 = stablehlo.reduce(%5981 init: %cst_1090) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %5983 = stablehlo.convert %5982 : (tensor<f32>) -> tensor<bf16>
    %5984 = stablehlo.add %5979, %5983 : tensor<bf16>
    %5985 = stablehlo.multiply %3645, %3645 : tensor<2016x576xbf16>
    %5986 = stablehlo.convert %5985 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1091 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5987 = stablehlo.reduce(%5986 init: %cst_1091) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %5988 = stablehlo.convert %5987 : (tensor<f32>) -> tensor<bf16>
    %5989 = stablehlo.add %5984, %5988 : tensor<bf16>
    %5990 = stablehlo.multiply %3650, %3650 : tensor<576xbf16>
    %5991 = stablehlo.convert %5990 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1092 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5992 = stablehlo.reduce(%5991 init: %cst_1092) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %5993 = stablehlo.convert %5992 : (tensor<f32>) -> tensor<bf16>
    %5994 = stablehlo.add %5989, %5993 : tensor<bf16>
    %5995 = stablehlo.multiply %3655, %3655 : tensor<576xbf16>
    %5996 = stablehlo.convert %5995 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1093 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5997 = stablehlo.reduce(%5996 init: %cst_1093) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %5998 = stablehlo.convert %5997 : (tensor<f32>) -> tensor<bf16>
    %5999 = stablehlo.add %5994, %5998 : tensor<bf16>
    %6000 = stablehlo.multiply %3660, %3660 : tensor<576x9x128xbf16>
    %6001 = stablehlo.convert %6000 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1094 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6002 = stablehlo.reduce(%6001 init: %cst_1094) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6003 = stablehlo.convert %6002 : (tensor<f32>) -> tensor<bf16>
    %6004 = stablehlo.add %5999, %6003 : tensor<bf16>
    %6005 = stablehlo.multiply %3665, %3665 : tensor<9x128x576xbf16>
    %6006 = stablehlo.convert %6005 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1095 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6007 = stablehlo.reduce(%6006 init: %cst_1095) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6008 = stablehlo.convert %6007 : (tensor<f32>) -> tensor<bf16>
    %6009 = stablehlo.add %6004, %6008 : tensor<bf16>
    %6010 = stablehlo.multiply %3670, %3670 : tensor<576x9x128xbf16>
    %6011 = stablehlo.convert %6010 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1096 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6012 = stablehlo.reduce(%6011 init: %cst_1096) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6013 = stablehlo.convert %6012 : (tensor<f32>) -> tensor<bf16>
    %6014 = stablehlo.add %6009, %6013 : tensor<bf16>
    %6015 = stablehlo.multiply %3675, %3675 : tensor<576x9x128xbf16>
    %6016 = stablehlo.convert %6015 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1097 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6017 = stablehlo.reduce(%6016 init: %cst_1097) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6018 = stablehlo.convert %6017 : (tensor<f32>) -> tensor<bf16>
    %6019 = stablehlo.add %6014, %6018 : tensor<bf16>
    %6020 = stablehlo.multiply %3680, %3680 : tensor<576x2016xbf16>
    %6021 = stablehlo.convert %6020 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1098 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6022 = stablehlo.reduce(%6021 init: %cst_1098) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6023 = stablehlo.convert %6022 : (tensor<f32>) -> tensor<bf16>
    %6024 = stablehlo.add %6019, %6023 : tensor<bf16>
    %6025 = stablehlo.multiply %3685, %3685 : tensor<576x2016xbf16>
    %6026 = stablehlo.convert %6025 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1099 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6027 = stablehlo.reduce(%6026 init: %cst_1099) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6028 = stablehlo.convert %6027 : (tensor<f32>) -> tensor<bf16>
    %6029 = stablehlo.add %6024, %6028 : tensor<bf16>
    %6030 = stablehlo.multiply %3690, %3690 : tensor<2016x576xbf16>
    %6031 = stablehlo.convert %6030 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6032 = stablehlo.reduce(%6031 init: %cst_1100) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6033 = stablehlo.convert %6032 : (tensor<f32>) -> tensor<bf16>
    %6034 = stablehlo.add %6029, %6033 : tensor<bf16>
    %6035 = stablehlo.multiply %3695, %3695 : tensor<576xbf16>
    %6036 = stablehlo.convert %6035 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1101 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6037 = stablehlo.reduce(%6036 init: %cst_1101) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6038 = stablehlo.convert %6037 : (tensor<f32>) -> tensor<bf16>
    %6039 = stablehlo.add %6034, %6038 : tensor<bf16>
    %6040 = stablehlo.multiply %3700, %3700 : tensor<576xbf16>
    %6041 = stablehlo.convert %6040 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1102 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6042 = stablehlo.reduce(%6041 init: %cst_1102) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6043 = stablehlo.convert %6042 : (tensor<f32>) -> tensor<bf16>
    %6044 = stablehlo.add %6039, %6043 : tensor<bf16>
    %6045 = stablehlo.multiply %3705, %3705 : tensor<576x9x128xbf16>
    %6046 = stablehlo.convert %6045 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1103 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6047 = stablehlo.reduce(%6046 init: %cst_1103) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6048 = stablehlo.convert %6047 : (tensor<f32>) -> tensor<bf16>
    %6049 = stablehlo.add %6044, %6048 : tensor<bf16>
    %6050 = stablehlo.multiply %3710, %3710 : tensor<9x128x576xbf16>
    %6051 = stablehlo.convert %6050 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6052 = stablehlo.reduce(%6051 init: %cst_1104) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6053 = stablehlo.convert %6052 : (tensor<f32>) -> tensor<bf16>
    %6054 = stablehlo.add %6049, %6053 : tensor<bf16>
    %6055 = stablehlo.multiply %3715, %3715 : tensor<576x9x128xbf16>
    %6056 = stablehlo.convert %6055 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1105 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6057 = stablehlo.reduce(%6056 init: %cst_1105) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6058 = stablehlo.convert %6057 : (tensor<f32>) -> tensor<bf16>
    %6059 = stablehlo.add %6054, %6058 : tensor<bf16>
    %6060 = stablehlo.multiply %3720, %3720 : tensor<576x9x128xbf16>
    %6061 = stablehlo.convert %6060 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6062 = stablehlo.reduce(%6061 init: %cst_1106) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6063 = stablehlo.convert %6062 : (tensor<f32>) -> tensor<bf16>
    %6064 = stablehlo.add %6059, %6063 : tensor<bf16>
    %6065 = stablehlo.multiply %3725, %3725 : tensor<576x2016xbf16>
    %6066 = stablehlo.convert %6065 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1107 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6067 = stablehlo.reduce(%6066 init: %cst_1107) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6068 = stablehlo.convert %6067 : (tensor<f32>) -> tensor<bf16>
    %6069 = stablehlo.add %6064, %6068 : tensor<bf16>
    %6070 = stablehlo.multiply %3730, %3730 : tensor<576x2016xbf16>
    %6071 = stablehlo.convert %6070 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1108 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6072 = stablehlo.reduce(%6071 init: %cst_1108) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6073 = stablehlo.convert %6072 : (tensor<f32>) -> tensor<bf16>
    %6074 = stablehlo.add %6069, %6073 : tensor<bf16>
    %6075 = stablehlo.multiply %3735, %3735 : tensor<2016x576xbf16>
    %6076 = stablehlo.convert %6075 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1109 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6077 = stablehlo.reduce(%6076 init: %cst_1109) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6078 = stablehlo.convert %6077 : (tensor<f32>) -> tensor<bf16>
    %6079 = stablehlo.add %6074, %6078 : tensor<bf16>
    %6080 = stablehlo.multiply %3740, %3740 : tensor<576xbf16>
    %6081 = stablehlo.convert %6080 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1110 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6082 = stablehlo.reduce(%6081 init: %cst_1110) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6083 = stablehlo.convert %6082 : (tensor<f32>) -> tensor<bf16>
    %6084 = stablehlo.add %6079, %6083 : tensor<bf16>
    %6085 = stablehlo.multiply %3745, %3745 : tensor<576xbf16>
    %6086 = stablehlo.convert %6085 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1111 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6087 = stablehlo.reduce(%6086 init: %cst_1111) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6088 = stablehlo.convert %6087 : (tensor<f32>) -> tensor<bf16>
    %6089 = stablehlo.add %6084, %6088 : tensor<bf16>
    %6090 = stablehlo.multiply %3750, %3750 : tensor<576x9x128xbf16>
    %6091 = stablehlo.convert %6090 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1112 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6092 = stablehlo.reduce(%6091 init: %cst_1112) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6093 = stablehlo.convert %6092 : (tensor<f32>) -> tensor<bf16>
    %6094 = stablehlo.add %6089, %6093 : tensor<bf16>
    %6095 = stablehlo.multiply %3755, %3755 : tensor<9x128x576xbf16>
    %6096 = stablehlo.convert %6095 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1113 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6097 = stablehlo.reduce(%6096 init: %cst_1113) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6098 = stablehlo.convert %6097 : (tensor<f32>) -> tensor<bf16>
    %6099 = stablehlo.add %6094, %6098 : tensor<bf16>
    %6100 = stablehlo.multiply %3760, %3760 : tensor<576x9x128xbf16>
    %6101 = stablehlo.convert %6100 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6102 = stablehlo.reduce(%6101 init: %cst_1114) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6103 = stablehlo.convert %6102 : (tensor<f32>) -> tensor<bf16>
    %6104 = stablehlo.add %6099, %6103 : tensor<bf16>
    %6105 = stablehlo.multiply %3765, %3765 : tensor<576x9x128xbf16>
    %6106 = stablehlo.convert %6105 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1115 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6107 = stablehlo.reduce(%6106 init: %cst_1115) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6108 = stablehlo.convert %6107 : (tensor<f32>) -> tensor<bf16>
    %6109 = stablehlo.add %6104, %6108 : tensor<bf16>
    %6110 = stablehlo.multiply %3770, %3770 : tensor<576x2016xbf16>
    %6111 = stablehlo.convert %6110 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1116 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6112 = stablehlo.reduce(%6111 init: %cst_1116) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6113 = stablehlo.convert %6112 : (tensor<f32>) -> tensor<bf16>
    %6114 = stablehlo.add %6109, %6113 : tensor<bf16>
    %6115 = stablehlo.multiply %3775, %3775 : tensor<576x2016xbf16>
    %6116 = stablehlo.convert %6115 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1117 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6117 = stablehlo.reduce(%6116 init: %cst_1117) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6118 = stablehlo.convert %6117 : (tensor<f32>) -> tensor<bf16>
    %6119 = stablehlo.add %6114, %6118 : tensor<bf16>
    %6120 = stablehlo.multiply %3780, %3780 : tensor<2016x576xbf16>
    %6121 = stablehlo.convert %6120 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6122 = stablehlo.reduce(%6121 init: %cst_1118) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6123 = stablehlo.convert %6122 : (tensor<f32>) -> tensor<bf16>
    %6124 = stablehlo.add %6119, %6123 : tensor<bf16>
    %6125 = stablehlo.multiply %3785, %3785 : tensor<576xbf16>
    %6126 = stablehlo.convert %6125 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1119 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6127 = stablehlo.reduce(%6126 init: %cst_1119) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6128 = stablehlo.convert %6127 : (tensor<f32>) -> tensor<bf16>
    %6129 = stablehlo.add %6124, %6128 : tensor<bf16>
    %6130 = stablehlo.multiply %3790, %3790 : tensor<576xbf16>
    %6131 = stablehlo.convert %6130 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6132 = stablehlo.reduce(%6131 init: %cst_1120) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6133 = stablehlo.convert %6132 : (tensor<f32>) -> tensor<bf16>
    %6134 = stablehlo.add %6129, %6133 : tensor<bf16>
    %6135 = stablehlo.multiply %3795, %3795 : tensor<576x9x128xbf16>
    %6136 = stablehlo.convert %6135 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6137 = stablehlo.reduce(%6136 init: %cst_1121) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6138 = stablehlo.convert %6137 : (tensor<f32>) -> tensor<bf16>
    %6139 = stablehlo.add %6134, %6138 : tensor<bf16>
    %6140 = stablehlo.multiply %3800, %3800 : tensor<9x128x576xbf16>
    %6141 = stablehlo.convert %6140 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1122 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6142 = stablehlo.reduce(%6141 init: %cst_1122) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6143 = stablehlo.convert %6142 : (tensor<f32>) -> tensor<bf16>
    %6144 = stablehlo.add %6139, %6143 : tensor<bf16>
    %6145 = stablehlo.multiply %3805, %3805 : tensor<576x9x128xbf16>
    %6146 = stablehlo.convert %6145 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1123 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6147 = stablehlo.reduce(%6146 init: %cst_1123) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6148 = stablehlo.convert %6147 : (tensor<f32>) -> tensor<bf16>
    %6149 = stablehlo.add %6144, %6148 : tensor<bf16>
    %6150 = stablehlo.multiply %3810, %3810 : tensor<576x9x128xbf16>
    %6151 = stablehlo.convert %6150 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1124 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6152 = stablehlo.reduce(%6151 init: %cst_1124) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6153 = stablehlo.convert %6152 : (tensor<f32>) -> tensor<bf16>
    %6154 = stablehlo.add %6149, %6153 : tensor<bf16>
    %6155 = stablehlo.multiply %3815, %3815 : tensor<576x2016xbf16>
    %6156 = stablehlo.convert %6155 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1125 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6157 = stablehlo.reduce(%6156 init: %cst_1125) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6158 = stablehlo.convert %6157 : (tensor<f32>) -> tensor<bf16>
    %6159 = stablehlo.add %6154, %6158 : tensor<bf16>
    %6160 = stablehlo.multiply %3820, %3820 : tensor<576x2016xbf16>
    %6161 = stablehlo.convert %6160 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6162 = stablehlo.reduce(%6161 init: %cst_1126) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6163 = stablehlo.convert %6162 : (tensor<f32>) -> tensor<bf16>
    %6164 = stablehlo.add %6159, %6163 : tensor<bf16>
    %6165 = stablehlo.multiply %3825, %3825 : tensor<2016x576xbf16>
    %6166 = stablehlo.convert %6165 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1127 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6167 = stablehlo.reduce(%6166 init: %cst_1127) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6168 = stablehlo.convert %6167 : (tensor<f32>) -> tensor<bf16>
    %6169 = stablehlo.add %6164, %6168 : tensor<bf16>
    %6170 = stablehlo.multiply %3830, %3830 : tensor<576xbf16>
    %6171 = stablehlo.convert %6170 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1128 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6172 = stablehlo.reduce(%6171 init: %cst_1128) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6173 = stablehlo.convert %6172 : (tensor<f32>) -> tensor<bf16>
    %6174 = stablehlo.add %6169, %6173 : tensor<bf16>
    %6175 = stablehlo.multiply %3835, %3835 : tensor<576xbf16>
    %6176 = stablehlo.convert %6175 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1129 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6177 = stablehlo.reduce(%6176 init: %cst_1129) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6178 = stablehlo.convert %6177 : (tensor<f32>) -> tensor<bf16>
    %6179 = stablehlo.add %6174, %6178 : tensor<bf16>
    %6180 = stablehlo.multiply %3840, %3840 : tensor<576x9x128xbf16>
    %6181 = stablehlo.convert %6180 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1130 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6182 = stablehlo.reduce(%6181 init: %cst_1130) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6183 = stablehlo.convert %6182 : (tensor<f32>) -> tensor<bf16>
    %6184 = stablehlo.add %6179, %6183 : tensor<bf16>
    %6185 = stablehlo.multiply %3845, %3845 : tensor<9x128x576xbf16>
    %6186 = stablehlo.convert %6185 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1131 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6187 = stablehlo.reduce(%6186 init: %cst_1131) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6188 = stablehlo.convert %6187 : (tensor<f32>) -> tensor<bf16>
    %6189 = stablehlo.add %6184, %6188 : tensor<bf16>
    %6190 = stablehlo.multiply %3850, %3850 : tensor<576x9x128xbf16>
    %6191 = stablehlo.convert %6190 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1132 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6192 = stablehlo.reduce(%6191 init: %cst_1132) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6193 = stablehlo.convert %6192 : (tensor<f32>) -> tensor<bf16>
    %6194 = stablehlo.add %6189, %6193 : tensor<bf16>
    %6195 = stablehlo.multiply %3855, %3855 : tensor<576x9x128xbf16>
    %6196 = stablehlo.convert %6195 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1133 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6197 = stablehlo.reduce(%6196 init: %cst_1133) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6198 = stablehlo.convert %6197 : (tensor<f32>) -> tensor<bf16>
    %6199 = stablehlo.add %6194, %6198 : tensor<bf16>
    %6200 = stablehlo.multiply %3860, %3860 : tensor<576x2016xbf16>
    %6201 = stablehlo.convert %6200 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1134 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6202 = stablehlo.reduce(%6201 init: %cst_1134) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6203 = stablehlo.convert %6202 : (tensor<f32>) -> tensor<bf16>
    %6204 = stablehlo.add %6199, %6203 : tensor<bf16>
    %6205 = stablehlo.multiply %3865, %3865 : tensor<576x2016xbf16>
    %6206 = stablehlo.convert %6205 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1135 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6207 = stablehlo.reduce(%6206 init: %cst_1135) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6208 = stablehlo.convert %6207 : (tensor<f32>) -> tensor<bf16>
    %6209 = stablehlo.add %6204, %6208 : tensor<bf16>
    %6210 = stablehlo.multiply %3870, %3870 : tensor<2016x576xbf16>
    %6211 = stablehlo.convert %6210 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1136 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6212 = stablehlo.reduce(%6211 init: %cst_1136) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6213 = stablehlo.convert %6212 : (tensor<f32>) -> tensor<bf16>
    %6214 = stablehlo.add %6209, %6213 : tensor<bf16>
    %6215 = stablehlo.multiply %3875, %3875 : tensor<576xbf16>
    %6216 = stablehlo.convert %6215 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1137 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6217 = stablehlo.reduce(%6216 init: %cst_1137) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6218 = stablehlo.convert %6217 : (tensor<f32>) -> tensor<bf16>
    %6219 = stablehlo.add %6214, %6218 : tensor<bf16>
    %6220 = stablehlo.multiply %3880, %3880 : tensor<576xbf16>
    %6221 = stablehlo.convert %6220 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1138 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6222 = stablehlo.reduce(%6221 init: %cst_1138) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6223 = stablehlo.convert %6222 : (tensor<f32>) -> tensor<bf16>
    %6224 = stablehlo.add %6219, %6223 : tensor<bf16>
    %6225 = stablehlo.multiply %3885, %3885 : tensor<576x9x128xbf16>
    %6226 = stablehlo.convert %6225 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1139 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6227 = stablehlo.reduce(%6226 init: %cst_1139) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6228 = stablehlo.convert %6227 : (tensor<f32>) -> tensor<bf16>
    %6229 = stablehlo.add %6224, %6228 : tensor<bf16>
    %6230 = stablehlo.multiply %3890, %3890 : tensor<9x128x576xbf16>
    %6231 = stablehlo.convert %6230 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1140 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6232 = stablehlo.reduce(%6231 init: %cst_1140) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6233 = stablehlo.convert %6232 : (tensor<f32>) -> tensor<bf16>
    %6234 = stablehlo.add %6229, %6233 : tensor<bf16>
    %6235 = stablehlo.multiply %3895, %3895 : tensor<576x9x128xbf16>
    %6236 = stablehlo.convert %6235 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6237 = stablehlo.reduce(%6236 init: %cst_1141) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6238 = stablehlo.convert %6237 : (tensor<f32>) -> tensor<bf16>
    %6239 = stablehlo.add %6234, %6238 : tensor<bf16>
    %6240 = stablehlo.multiply %3900, %3900 : tensor<576x9x128xbf16>
    %6241 = stablehlo.convert %6240 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6242 = stablehlo.reduce(%6241 init: %cst_1142) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6243 = stablehlo.convert %6242 : (tensor<f32>) -> tensor<bf16>
    %6244 = stablehlo.add %6239, %6243 : tensor<bf16>
    %6245 = stablehlo.multiply %3905, %3905 : tensor<576x128256xbf16>
    %6246 = stablehlo.convert %6245 : (tensor<576x128256xbf16>) -> tensor<576x128256xf32>
    %cst_1143 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6247 = stablehlo.reduce(%6246 init: %cst_1143) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x128256xf32>, tensor<f32>) -> tensor<f32>
    %6248 = stablehlo.convert %6247 : (tensor<f32>) -> tensor<bf16>
    %6249 = stablehlo.add %6244, %6248 : tensor<bf16>
    %6250 = stablehlo.multiply %3910, %3910 : tensor<128256x576xbf16>
    %6251 = stablehlo.convert %6250 : (tensor<128256x576xbf16>) -> tensor<128256x576xf32>
    %cst_1144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6252 = stablehlo.reduce(%6251 init: %cst_1144) applies stablehlo.add across dimensions = [0, 1] : (tensor<128256x576xf32>, tensor<f32>) -> tensor<f32>
    %6253 = stablehlo.convert %6252 : (tensor<f32>) -> tensor<bf16>
    %6254 = stablehlo.add %6249, %6253 : tensor<bf16>
    %6255 = stablehlo.sqrt %6254 : tensor<bf16>
    %6256 = stablehlo.multiply %1236, %1236 : tensor<576xbf16>
    %6257 = stablehlo.convert %6256 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1145 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6258 = stablehlo.reduce(%6257 init: %cst_1145) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6259 = stablehlo.convert %6258 : (tensor<f32>) -> tensor<bf16>
    %cst_1146 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %6260 = stablehlo.add %cst_1146, %6259 : tensor<bf16>
    %6261 = stablehlo.multiply %3112, %3112 : tensor<576x2016xbf16>
    %6262 = stablehlo.convert %6261 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1147 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6263 = stablehlo.reduce(%6262 init: %cst_1147) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6264 = stablehlo.convert %6263 : (tensor<f32>) -> tensor<bf16>
    %6265 = stablehlo.add %6260, %6264 : tensor<bf16>
    %6266 = stablehlo.multiply %3108, %3108 : tensor<576x2016xbf16>
    %6267 = stablehlo.convert %6266 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1148 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6268 = stablehlo.reduce(%6267 init: %cst_1148) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6269 = stablehlo.convert %6268 : (tensor<f32>) -> tensor<bf16>
    %6270 = stablehlo.add %6265, %6269 : tensor<bf16>
    %6271 = stablehlo.multiply %3102, %3102 : tensor<2016x576xbf16>
    %6272 = stablehlo.convert %6271 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1149 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6273 = stablehlo.reduce(%6272 init: %cst_1149) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6274 = stablehlo.convert %6273 : (tensor<f32>) -> tensor<bf16>
    %6275 = stablehlo.add %6270, %6274 : tensor<bf16>
    %6276 = stablehlo.multiply %3121, %3121 : tensor<576xbf16>
    %6277 = stablehlo.convert %6276 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1150 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6278 = stablehlo.reduce(%6277 init: %cst_1150) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6279 = stablehlo.convert %6278 : (tensor<f32>) -> tensor<bf16>
    %6280 = stablehlo.add %6275, %6279 : tensor<bf16>
    %6281 = stablehlo.multiply %3229, %3229 : tensor<576xbf16>
    %6282 = stablehlo.convert %6281 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1151 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6283 = stablehlo.reduce(%6282 init: %cst_1151) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6284 = stablehlo.convert %6283 : (tensor<f32>) -> tensor<bf16>
    %6285 = stablehlo.add %6280, %6284 : tensor<bf16>
    %6286 = stablehlo.multiply %3214, %3214 : tensor<576x9x128xbf16>
    %6287 = stablehlo.convert %6286 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1152 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6288 = stablehlo.reduce(%6287 init: %cst_1152) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6289 = stablehlo.convert %6288 : (tensor<f32>) -> tensor<bf16>
    %6290 = stablehlo.add %6285, %6289 : tensor<bf16>
    %6291 = stablehlo.multiply %3139, %3139 : tensor<9x128x576xbf16>
    %6292 = stablehlo.convert %6291 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1153 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6293 = stablehlo.reduce(%6292 init: %cst_1153) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6294 = stablehlo.convert %6293 : (tensor<f32>) -> tensor<bf16>
    %6295 = stablehlo.add %6290, %6294 : tensor<bf16>
    %6296 = stablehlo.multiply %3218, %3218 : tensor<576x9x128xbf16>
    %6297 = stablehlo.convert %6296 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1154 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6298 = stablehlo.reduce(%6297 init: %cst_1154) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6299 = stablehlo.convert %6298 : (tensor<f32>) -> tensor<bf16>
    %6300 = stablehlo.add %6295, %6299 : tensor<bf16>
    %6301 = stablehlo.multiply %3211, %3211 : tensor<576x9x128xbf16>
    %6302 = stablehlo.convert %6301 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6303 = stablehlo.reduce(%6302 init: %cst_1155) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6304 = stablehlo.convert %6303 : (tensor<f32>) -> tensor<bf16>
    %6305 = stablehlo.add %6300, %6304 : tensor<bf16>
    %6306 = stablehlo.multiply %2827, %2827 : tensor<576x2016xbf16>
    %6307 = stablehlo.convert %6306 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1156 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6308 = stablehlo.reduce(%6307 init: %cst_1156) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6309 = stablehlo.convert %6308 : (tensor<f32>) -> tensor<bf16>
    %6310 = stablehlo.add %6305, %6309 : tensor<bf16>
    %6311 = stablehlo.multiply %2823, %2823 : tensor<576x2016xbf16>
    %6312 = stablehlo.convert %6311 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6313 = stablehlo.reduce(%6312 init: %cst_1157) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6314 = stablehlo.convert %6313 : (tensor<f32>) -> tensor<bf16>
    %6315 = stablehlo.add %6310, %6314 : tensor<bf16>
    %6316 = stablehlo.multiply %2817, %2817 : tensor<2016x576xbf16>
    %6317 = stablehlo.convert %6316 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1158 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6318 = stablehlo.reduce(%6317 init: %cst_1158) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6319 = stablehlo.convert %6318 : (tensor<f32>) -> tensor<bf16>
    %6320 = stablehlo.add %6315, %6319 : tensor<bf16>
    %6321 = stablehlo.multiply %2836, %2836 : tensor<576xbf16>
    %6322 = stablehlo.convert %6321 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1159 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6323 = stablehlo.reduce(%6322 init: %cst_1159) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6324 = stablehlo.convert %6323 : (tensor<f32>) -> tensor<bf16>
    %6325 = stablehlo.add %6320, %6324 : tensor<bf16>
    %6326 = stablehlo.multiply %2944, %2944 : tensor<576xbf16>
    %6327 = stablehlo.convert %6326 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1160 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6328 = stablehlo.reduce(%6327 init: %cst_1160) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6329 = stablehlo.convert %6328 : (tensor<f32>) -> tensor<bf16>
    %6330 = stablehlo.add %6325, %6329 : tensor<bf16>
    %6331 = stablehlo.multiply %2929, %2929 : tensor<576x9x128xbf16>
    %6332 = stablehlo.convert %6331 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1161 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6333 = stablehlo.reduce(%6332 init: %cst_1161) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6334 = stablehlo.convert %6333 : (tensor<f32>) -> tensor<bf16>
    %6335 = stablehlo.add %6330, %6334 : tensor<bf16>
    %6336 = stablehlo.multiply %2854, %2854 : tensor<9x128x576xbf16>
    %6337 = stablehlo.convert %6336 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1162 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6338 = stablehlo.reduce(%6337 init: %cst_1162) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6339 = stablehlo.convert %6338 : (tensor<f32>) -> tensor<bf16>
    %6340 = stablehlo.add %6335, %6339 : tensor<bf16>
    %6341 = stablehlo.multiply %2933, %2933 : tensor<576x9x128xbf16>
    %6342 = stablehlo.convert %6341 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6343 = stablehlo.reduce(%6342 init: %cst_1163) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6344 = stablehlo.convert %6343 : (tensor<f32>) -> tensor<bf16>
    %6345 = stablehlo.add %6340, %6344 : tensor<bf16>
    %6346 = stablehlo.multiply %2926, %2926 : tensor<576x9x128xbf16>
    %6347 = stablehlo.convert %6346 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6348 = stablehlo.reduce(%6347 init: %cst_1164) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6349 = stablehlo.convert %6348 : (tensor<f32>) -> tensor<bf16>
    %6350 = stablehlo.add %6345, %6349 : tensor<bf16>
    %6351 = stablehlo.multiply %2542, %2542 : tensor<576x2016xbf16>
    %6352 = stablehlo.convert %6351 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1165 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6353 = stablehlo.reduce(%6352 init: %cst_1165) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6354 = stablehlo.convert %6353 : (tensor<f32>) -> tensor<bf16>
    %6355 = stablehlo.add %6350, %6354 : tensor<bf16>
    %6356 = stablehlo.multiply %2538, %2538 : tensor<576x2016xbf16>
    %6357 = stablehlo.convert %6356 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1166 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6358 = stablehlo.reduce(%6357 init: %cst_1166) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6359 = stablehlo.convert %6358 : (tensor<f32>) -> tensor<bf16>
    %6360 = stablehlo.add %6355, %6359 : tensor<bf16>
    %6361 = stablehlo.multiply %2532, %2532 : tensor<2016x576xbf16>
    %6362 = stablehlo.convert %6361 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1167 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6363 = stablehlo.reduce(%6362 init: %cst_1167) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6364 = stablehlo.convert %6363 : (tensor<f32>) -> tensor<bf16>
    %6365 = stablehlo.add %6360, %6364 : tensor<bf16>
    %6366 = stablehlo.multiply %2551, %2551 : tensor<576xbf16>
    %6367 = stablehlo.convert %6366 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1168 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6368 = stablehlo.reduce(%6367 init: %cst_1168) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6369 = stablehlo.convert %6368 : (tensor<f32>) -> tensor<bf16>
    %6370 = stablehlo.add %6365, %6369 : tensor<bf16>
    %6371 = stablehlo.multiply %2659, %2659 : tensor<576xbf16>
    %6372 = stablehlo.convert %6371 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1169 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6373 = stablehlo.reduce(%6372 init: %cst_1169) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6374 = stablehlo.convert %6373 : (tensor<f32>) -> tensor<bf16>
    %6375 = stablehlo.add %6370, %6374 : tensor<bf16>
    %6376 = stablehlo.multiply %2644, %2644 : tensor<576x9x128xbf16>
    %6377 = stablehlo.convert %6376 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1170 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6378 = stablehlo.reduce(%6377 init: %cst_1170) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6379 = stablehlo.convert %6378 : (tensor<f32>) -> tensor<bf16>
    %6380 = stablehlo.add %6375, %6379 : tensor<bf16>
    %6381 = stablehlo.multiply %2569, %2569 : tensor<9x128x576xbf16>
    %6382 = stablehlo.convert %6381 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6383 = stablehlo.reduce(%6382 init: %cst_1171) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6384 = stablehlo.convert %6383 : (tensor<f32>) -> tensor<bf16>
    %6385 = stablehlo.add %6380, %6384 : tensor<bf16>
    %6386 = stablehlo.multiply %2648, %2648 : tensor<576x9x128xbf16>
    %6387 = stablehlo.convert %6386 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1172 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6388 = stablehlo.reduce(%6387 init: %cst_1172) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6389 = stablehlo.convert %6388 : (tensor<f32>) -> tensor<bf16>
    %6390 = stablehlo.add %6385, %6389 : tensor<bf16>
    %6391 = stablehlo.multiply %2641, %2641 : tensor<576x9x128xbf16>
    %6392 = stablehlo.convert %6391 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1173 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6393 = stablehlo.reduce(%6392 init: %cst_1173) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6394 = stablehlo.convert %6393 : (tensor<f32>) -> tensor<bf16>
    %6395 = stablehlo.add %6390, %6394 : tensor<bf16>
    %6396 = stablehlo.multiply %2257, %2257 : tensor<576x2016xbf16>
    %6397 = stablehlo.convert %6396 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1174 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6398 = stablehlo.reduce(%6397 init: %cst_1174) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6399 = stablehlo.convert %6398 : (tensor<f32>) -> tensor<bf16>
    %6400 = stablehlo.add %6395, %6399 : tensor<bf16>
    %6401 = stablehlo.multiply %2253, %2253 : tensor<576x2016xbf16>
    %6402 = stablehlo.convert %6401 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1175 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6403 = stablehlo.reduce(%6402 init: %cst_1175) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6404 = stablehlo.convert %6403 : (tensor<f32>) -> tensor<bf16>
    %6405 = stablehlo.add %6400, %6404 : tensor<bf16>
    %6406 = stablehlo.multiply %2247, %2247 : tensor<2016x576xbf16>
    %6407 = stablehlo.convert %6406 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1176 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6408 = stablehlo.reduce(%6407 init: %cst_1176) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6409 = stablehlo.convert %6408 : (tensor<f32>) -> tensor<bf16>
    %6410 = stablehlo.add %6405, %6409 : tensor<bf16>
    %6411 = stablehlo.multiply %2266, %2266 : tensor<576xbf16>
    %6412 = stablehlo.convert %6411 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1177 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6413 = stablehlo.reduce(%6412 init: %cst_1177) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6414 = stablehlo.convert %6413 : (tensor<f32>) -> tensor<bf16>
    %6415 = stablehlo.add %6410, %6414 : tensor<bf16>
    %6416 = stablehlo.multiply %2374, %2374 : tensor<576xbf16>
    %6417 = stablehlo.convert %6416 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1178 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6418 = stablehlo.reduce(%6417 init: %cst_1178) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6419 = stablehlo.convert %6418 : (tensor<f32>) -> tensor<bf16>
    %6420 = stablehlo.add %6415, %6419 : tensor<bf16>
    %6421 = stablehlo.multiply %2359, %2359 : tensor<576x9x128xbf16>
    %6422 = stablehlo.convert %6421 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1179 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6423 = stablehlo.reduce(%6422 init: %cst_1179) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6424 = stablehlo.convert %6423 : (tensor<f32>) -> tensor<bf16>
    %6425 = stablehlo.add %6420, %6424 : tensor<bf16>
    %6426 = stablehlo.multiply %2284, %2284 : tensor<9x128x576xbf16>
    %6427 = stablehlo.convert %6426 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1180 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6428 = stablehlo.reduce(%6427 init: %cst_1180) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6429 = stablehlo.convert %6428 : (tensor<f32>) -> tensor<bf16>
    %6430 = stablehlo.add %6425, %6429 : tensor<bf16>
    %6431 = stablehlo.multiply %2363, %2363 : tensor<576x9x128xbf16>
    %6432 = stablehlo.convert %6431 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1181 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6433 = stablehlo.reduce(%6432 init: %cst_1181) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6434 = stablehlo.convert %6433 : (tensor<f32>) -> tensor<bf16>
    %6435 = stablehlo.add %6430, %6434 : tensor<bf16>
    %6436 = stablehlo.multiply %2356, %2356 : tensor<576x9x128xbf16>
    %6437 = stablehlo.convert %6436 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1182 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6438 = stablehlo.reduce(%6437 init: %cst_1182) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6439 = stablehlo.convert %6438 : (tensor<f32>) -> tensor<bf16>
    %6440 = stablehlo.add %6435, %6439 : tensor<bf16>
    %6441 = stablehlo.multiply %1972, %1972 : tensor<576x2016xbf16>
    %6442 = stablehlo.convert %6441 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1183 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6443 = stablehlo.reduce(%6442 init: %cst_1183) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6444 = stablehlo.convert %6443 : (tensor<f32>) -> tensor<bf16>
    %6445 = stablehlo.add %6440, %6444 : tensor<bf16>
    %6446 = stablehlo.multiply %1968, %1968 : tensor<576x2016xbf16>
    %6447 = stablehlo.convert %6446 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1184 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6448 = stablehlo.reduce(%6447 init: %cst_1184) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6449 = stablehlo.convert %6448 : (tensor<f32>) -> tensor<bf16>
    %6450 = stablehlo.add %6445, %6449 : tensor<bf16>
    %6451 = stablehlo.multiply %1962, %1962 : tensor<2016x576xbf16>
    %6452 = stablehlo.convert %6451 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1185 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6453 = stablehlo.reduce(%6452 init: %cst_1185) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6454 = stablehlo.convert %6453 : (tensor<f32>) -> tensor<bf16>
    %6455 = stablehlo.add %6450, %6454 : tensor<bf16>
    %6456 = stablehlo.multiply %1981, %1981 : tensor<576xbf16>
    %6457 = stablehlo.convert %6456 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1186 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6458 = stablehlo.reduce(%6457 init: %cst_1186) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6459 = stablehlo.convert %6458 : (tensor<f32>) -> tensor<bf16>
    %6460 = stablehlo.add %6455, %6459 : tensor<bf16>
    %6461 = stablehlo.multiply %2089, %2089 : tensor<576xbf16>
    %6462 = stablehlo.convert %6461 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1187 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6463 = stablehlo.reduce(%6462 init: %cst_1187) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6464 = stablehlo.convert %6463 : (tensor<f32>) -> tensor<bf16>
    %6465 = stablehlo.add %6460, %6464 : tensor<bf16>
    %6466 = stablehlo.multiply %2074, %2074 : tensor<576x9x128xbf16>
    %6467 = stablehlo.convert %6466 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1188 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6468 = stablehlo.reduce(%6467 init: %cst_1188) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6469 = stablehlo.convert %6468 : (tensor<f32>) -> tensor<bf16>
    %6470 = stablehlo.add %6465, %6469 : tensor<bf16>
    %6471 = stablehlo.multiply %1999, %1999 : tensor<9x128x576xbf16>
    %6472 = stablehlo.convert %6471 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1189 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6473 = stablehlo.reduce(%6472 init: %cst_1189) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6474 = stablehlo.convert %6473 : (tensor<f32>) -> tensor<bf16>
    %6475 = stablehlo.add %6470, %6474 : tensor<bf16>
    %6476 = stablehlo.multiply %2078, %2078 : tensor<576x9x128xbf16>
    %6477 = stablehlo.convert %6476 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1190 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6478 = stablehlo.reduce(%6477 init: %cst_1190) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6479 = stablehlo.convert %6478 : (tensor<f32>) -> tensor<bf16>
    %6480 = stablehlo.add %6475, %6479 : tensor<bf16>
    %6481 = stablehlo.multiply %2071, %2071 : tensor<576x9x128xbf16>
    %6482 = stablehlo.convert %6481 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1191 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6483 = stablehlo.reduce(%6482 init: %cst_1191) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6484 = stablehlo.convert %6483 : (tensor<f32>) -> tensor<bf16>
    %6485 = stablehlo.add %6480, %6484 : tensor<bf16>
    %6486 = stablehlo.multiply %1687, %1687 : tensor<576x2016xbf16>
    %6487 = stablehlo.convert %6486 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1192 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6488 = stablehlo.reduce(%6487 init: %cst_1192) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6489 = stablehlo.convert %6488 : (tensor<f32>) -> tensor<bf16>
    %6490 = stablehlo.add %6485, %6489 : tensor<bf16>
    %6491 = stablehlo.multiply %1683, %1683 : tensor<576x2016xbf16>
    %6492 = stablehlo.convert %6491 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1193 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6493 = stablehlo.reduce(%6492 init: %cst_1193) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6494 = stablehlo.convert %6493 : (tensor<f32>) -> tensor<bf16>
    %6495 = stablehlo.add %6490, %6494 : tensor<bf16>
    %6496 = stablehlo.multiply %1677, %1677 : tensor<2016x576xbf16>
    %6497 = stablehlo.convert %6496 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1194 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6498 = stablehlo.reduce(%6497 init: %cst_1194) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6499 = stablehlo.convert %6498 : (tensor<f32>) -> tensor<bf16>
    %6500 = stablehlo.add %6495, %6499 : tensor<bf16>
    %6501 = stablehlo.multiply %1696, %1696 : tensor<576xbf16>
    %6502 = stablehlo.convert %6501 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1195 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6503 = stablehlo.reduce(%6502 init: %cst_1195) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6504 = stablehlo.convert %6503 : (tensor<f32>) -> tensor<bf16>
    %6505 = stablehlo.add %6500, %6504 : tensor<bf16>
    %6506 = stablehlo.multiply %1804, %1804 : tensor<576xbf16>
    %6507 = stablehlo.convert %6506 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1196 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6508 = stablehlo.reduce(%6507 init: %cst_1196) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6509 = stablehlo.convert %6508 : (tensor<f32>) -> tensor<bf16>
    %6510 = stablehlo.add %6505, %6509 : tensor<bf16>
    %6511 = stablehlo.multiply %1789, %1789 : tensor<576x9x128xbf16>
    %6512 = stablehlo.convert %6511 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1197 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6513 = stablehlo.reduce(%6512 init: %cst_1197) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6514 = stablehlo.convert %6513 : (tensor<f32>) -> tensor<bf16>
    %6515 = stablehlo.add %6510, %6514 : tensor<bf16>
    %6516 = stablehlo.multiply %1714, %1714 : tensor<9x128x576xbf16>
    %6517 = stablehlo.convert %6516 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1198 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6518 = stablehlo.reduce(%6517 init: %cst_1198) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6519 = stablehlo.convert %6518 : (tensor<f32>) -> tensor<bf16>
    %6520 = stablehlo.add %6515, %6519 : tensor<bf16>
    %6521 = stablehlo.multiply %1793, %1793 : tensor<576x9x128xbf16>
    %6522 = stablehlo.convert %6521 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1199 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6523 = stablehlo.reduce(%6522 init: %cst_1199) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6524 = stablehlo.convert %6523 : (tensor<f32>) -> tensor<bf16>
    %6525 = stablehlo.add %6520, %6524 : tensor<bf16>
    %6526 = stablehlo.multiply %1786, %1786 : tensor<576x9x128xbf16>
    %6527 = stablehlo.convert %6526 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1200 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6528 = stablehlo.reduce(%6527 init: %cst_1200) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6529 = stablehlo.convert %6528 : (tensor<f32>) -> tensor<bf16>
    %6530 = stablehlo.add %6525, %6529 : tensor<bf16>
    %6531 = stablehlo.multiply %1402, %1402 : tensor<576x2016xbf16>
    %6532 = stablehlo.convert %6531 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6533 = stablehlo.reduce(%6532 init: %cst_1201) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6534 = stablehlo.convert %6533 : (tensor<f32>) -> tensor<bf16>
    %6535 = stablehlo.add %6530, %6534 : tensor<bf16>
    %6536 = stablehlo.multiply %1398, %1398 : tensor<576x2016xbf16>
    %6537 = stablehlo.convert %6536 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1202 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6538 = stablehlo.reduce(%6537 init: %cst_1202) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6539 = stablehlo.convert %6538 : (tensor<f32>) -> tensor<bf16>
    %6540 = stablehlo.add %6535, %6539 : tensor<bf16>
    %6541 = stablehlo.multiply %1392, %1392 : tensor<2016x576xbf16>
    %6542 = stablehlo.convert %6541 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1203 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6543 = stablehlo.reduce(%6542 init: %cst_1203) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6544 = stablehlo.convert %6543 : (tensor<f32>) -> tensor<bf16>
    %6545 = stablehlo.add %6540, %6544 : tensor<bf16>
    %6546 = stablehlo.multiply %1411, %1411 : tensor<576xbf16>
    %6547 = stablehlo.convert %6546 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1204 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6548 = stablehlo.reduce(%6547 init: %cst_1204) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6549 = stablehlo.convert %6548 : (tensor<f32>) -> tensor<bf16>
    %6550 = stablehlo.add %6545, %6549 : tensor<bf16>
    %6551 = stablehlo.multiply %1519, %1519 : tensor<576xbf16>
    %6552 = stablehlo.convert %6551 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1205 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6553 = stablehlo.reduce(%6552 init: %cst_1205) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6554 = stablehlo.convert %6553 : (tensor<f32>) -> tensor<bf16>
    %6555 = stablehlo.add %6550, %6554 : tensor<bf16>
    %6556 = stablehlo.multiply %1504, %1504 : tensor<576x9x128xbf16>
    %6557 = stablehlo.convert %6556 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1206 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6558 = stablehlo.reduce(%6557 init: %cst_1206) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6559 = stablehlo.convert %6558 : (tensor<f32>) -> tensor<bf16>
    %6560 = stablehlo.add %6555, %6559 : tensor<bf16>
    %6561 = stablehlo.multiply %1429, %1429 : tensor<9x128x576xbf16>
    %6562 = stablehlo.convert %6561 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1207 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6563 = stablehlo.reduce(%6562 init: %cst_1207) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6564 = stablehlo.convert %6563 : (tensor<f32>) -> tensor<bf16>
    %6565 = stablehlo.add %6560, %6564 : tensor<bf16>
    %6566 = stablehlo.multiply %1508, %1508 : tensor<576x9x128xbf16>
    %6567 = stablehlo.convert %6566 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1208 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6568 = stablehlo.reduce(%6567 init: %cst_1208) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6569 = stablehlo.convert %6568 : (tensor<f32>) -> tensor<bf16>
    %6570 = stablehlo.add %6565, %6569 : tensor<bf16>
    %6571 = stablehlo.multiply %1501, %1501 : tensor<576x9x128xbf16>
    %6572 = stablehlo.convert %6571 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1209 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6573 = stablehlo.reduce(%6572 init: %cst_1209) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6574 = stablehlo.convert %6573 : (tensor<f32>) -> tensor<bf16>
    %6575 = stablehlo.add %6570, %6574 : tensor<bf16>
    %6576 = stablehlo.multiply %1229, %1229 : tensor<576x128256xbf16>
    %6577 = stablehlo.convert %6576 : (tensor<576x128256xbf16>) -> tensor<576x128256xf32>
    %cst_1210 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6578 = stablehlo.reduce(%6577 init: %cst_1210) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x128256xf32>, tensor<f32>) -> tensor<f32>
    %6579 = stablehlo.convert %6578 : (tensor<f32>) -> tensor<bf16>
    %6580 = stablehlo.add %6575, %6579 : tensor<bf16>
    %6581 = stablehlo.multiply %3248, %3248 : tensor<128256x576xbf16>
    %6582 = stablehlo.convert %6581 : (tensor<128256x576xbf16>) -> tensor<128256x576xf32>
    %cst_1211 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6583 = stablehlo.reduce(%6582 init: %cst_1211) applies stablehlo.add across dimensions = [0, 1] : (tensor<128256x576xf32>, tensor<f32>) -> tensor<f32>
    %6584 = stablehlo.convert %6583 : (tensor<f32>) -> tensor<bf16>
    %6585 = stablehlo.add %6580, %6584 : tensor<bf16>
    %6586 = stablehlo.sqrt %6585 : tensor<bf16>
    %6587 = stablehlo.multiply %5858, %5858 : tensor<576xbf16>
    %6588 = stablehlo.convert %6587 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1212 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6589 = stablehlo.reduce(%6588 init: %cst_1212) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6590 = stablehlo.convert %6589 : (tensor<f32>) -> tensor<bf16>
    %cst_1213 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %6591 = stablehlo.add %cst_1213, %6590 : tensor<bf16>
    %6592 = stablehlo.multiply %5859, %5859 : tensor<576x2016xbf16>
    %6593 = stablehlo.convert %6592 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1214 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6594 = stablehlo.reduce(%6593 init: %cst_1214) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6595 = stablehlo.convert %6594 : (tensor<f32>) -> tensor<bf16>
    %6596 = stablehlo.add %6591, %6595 : tensor<bf16>
    %6597 = stablehlo.multiply %5860, %5860 : tensor<576x2016xbf16>
    %6598 = stablehlo.convert %6597 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1215 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6599 = stablehlo.reduce(%6598 init: %cst_1215) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6600 = stablehlo.convert %6599 : (tensor<f32>) -> tensor<bf16>
    %6601 = stablehlo.add %6596, %6600 : tensor<bf16>
    %6602 = stablehlo.multiply %5861, %5861 : tensor<2016x576xbf16>
    %6603 = stablehlo.convert %6602 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1216 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6604 = stablehlo.reduce(%6603 init: %cst_1216) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6605 = stablehlo.convert %6604 : (tensor<f32>) -> tensor<bf16>
    %6606 = stablehlo.add %6601, %6605 : tensor<bf16>
    %6607 = stablehlo.multiply %5862, %5862 : tensor<576xbf16>
    %6608 = stablehlo.convert %6607 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1217 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6609 = stablehlo.reduce(%6608 init: %cst_1217) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6610 = stablehlo.convert %6609 : (tensor<f32>) -> tensor<bf16>
    %6611 = stablehlo.add %6606, %6610 : tensor<bf16>
    %6612 = stablehlo.multiply %5863, %5863 : tensor<576xbf16>
    %6613 = stablehlo.convert %6612 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1218 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6614 = stablehlo.reduce(%6613 init: %cst_1218) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6615 = stablehlo.convert %6614 : (tensor<f32>) -> tensor<bf16>
    %6616 = stablehlo.add %6611, %6615 : tensor<bf16>
    %6617 = stablehlo.multiply %5864, %5864 : tensor<576x9x128xbf16>
    %6618 = stablehlo.convert %6617 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1219 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6619 = stablehlo.reduce(%6618 init: %cst_1219) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6620 = stablehlo.convert %6619 : (tensor<f32>) -> tensor<bf16>
    %6621 = stablehlo.add %6616, %6620 : tensor<bf16>
    %6622 = stablehlo.multiply %5865, %5865 : tensor<9x128x576xbf16>
    %6623 = stablehlo.convert %6622 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1220 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6624 = stablehlo.reduce(%6623 init: %cst_1220) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6625 = stablehlo.convert %6624 : (tensor<f32>) -> tensor<bf16>
    %6626 = stablehlo.add %6621, %6625 : tensor<bf16>
    %6627 = stablehlo.multiply %5866, %5866 : tensor<576x9x128xbf16>
    %6628 = stablehlo.convert %6627 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1221 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6629 = stablehlo.reduce(%6628 init: %cst_1221) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6630 = stablehlo.convert %6629 : (tensor<f32>) -> tensor<bf16>
    %6631 = stablehlo.add %6626, %6630 : tensor<bf16>
    %6632 = stablehlo.multiply %5867, %5867 : tensor<576x9x128xbf16>
    %6633 = stablehlo.convert %6632 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1222 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6634 = stablehlo.reduce(%6633 init: %cst_1222) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6635 = stablehlo.convert %6634 : (tensor<f32>) -> tensor<bf16>
    %6636 = stablehlo.add %6631, %6635 : tensor<bf16>
    %6637 = stablehlo.multiply %5868, %5868 : tensor<576x2016xbf16>
    %6638 = stablehlo.convert %6637 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1223 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6639 = stablehlo.reduce(%6638 init: %cst_1223) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6640 = stablehlo.convert %6639 : (tensor<f32>) -> tensor<bf16>
    %6641 = stablehlo.add %6636, %6640 : tensor<bf16>
    %6642 = stablehlo.multiply %5869, %5869 : tensor<576x2016xbf16>
    %6643 = stablehlo.convert %6642 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1224 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6644 = stablehlo.reduce(%6643 init: %cst_1224) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6645 = stablehlo.convert %6644 : (tensor<f32>) -> tensor<bf16>
    %6646 = stablehlo.add %6641, %6645 : tensor<bf16>
    %6647 = stablehlo.multiply %5870, %5870 : tensor<2016x576xbf16>
    %6648 = stablehlo.convert %6647 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1225 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6649 = stablehlo.reduce(%6648 init: %cst_1225) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6650 = stablehlo.convert %6649 : (tensor<f32>) -> tensor<bf16>
    %6651 = stablehlo.add %6646, %6650 : tensor<bf16>
    %6652 = stablehlo.multiply %5871, %5871 : tensor<576xbf16>
    %6653 = stablehlo.convert %6652 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1226 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6654 = stablehlo.reduce(%6653 init: %cst_1226) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6655 = stablehlo.convert %6654 : (tensor<f32>) -> tensor<bf16>
    %6656 = stablehlo.add %6651, %6655 : tensor<bf16>
    %6657 = stablehlo.multiply %5872, %5872 : tensor<576xbf16>
    %6658 = stablehlo.convert %6657 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1227 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6659 = stablehlo.reduce(%6658 init: %cst_1227) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6660 = stablehlo.convert %6659 : (tensor<f32>) -> tensor<bf16>
    %6661 = stablehlo.add %6656, %6660 : tensor<bf16>
    %6662 = stablehlo.multiply %5873, %5873 : tensor<576x9x128xbf16>
    %6663 = stablehlo.convert %6662 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1228 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6664 = stablehlo.reduce(%6663 init: %cst_1228) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6665 = stablehlo.convert %6664 : (tensor<f32>) -> tensor<bf16>
    %6666 = stablehlo.add %6661, %6665 : tensor<bf16>
    %6667 = stablehlo.multiply %5874, %5874 : tensor<9x128x576xbf16>
    %6668 = stablehlo.convert %6667 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1229 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6669 = stablehlo.reduce(%6668 init: %cst_1229) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6670 = stablehlo.convert %6669 : (tensor<f32>) -> tensor<bf16>
    %6671 = stablehlo.add %6666, %6670 : tensor<bf16>
    %6672 = stablehlo.multiply %5875, %5875 : tensor<576x9x128xbf16>
    %6673 = stablehlo.convert %6672 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1230 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6674 = stablehlo.reduce(%6673 init: %cst_1230) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6675 = stablehlo.convert %6674 : (tensor<f32>) -> tensor<bf16>
    %6676 = stablehlo.add %6671, %6675 : tensor<bf16>
    %6677 = stablehlo.multiply %5876, %5876 : tensor<576x9x128xbf16>
    %6678 = stablehlo.convert %6677 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1231 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6679 = stablehlo.reduce(%6678 init: %cst_1231) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6680 = stablehlo.convert %6679 : (tensor<f32>) -> tensor<bf16>
    %6681 = stablehlo.add %6676, %6680 : tensor<bf16>
    %6682 = stablehlo.multiply %5877, %5877 : tensor<576x2016xbf16>
    %6683 = stablehlo.convert %6682 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1232 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6684 = stablehlo.reduce(%6683 init: %cst_1232) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6685 = stablehlo.convert %6684 : (tensor<f32>) -> tensor<bf16>
    %6686 = stablehlo.add %6681, %6685 : tensor<bf16>
    %6687 = stablehlo.multiply %5878, %5878 : tensor<576x2016xbf16>
    %6688 = stablehlo.convert %6687 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1233 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6689 = stablehlo.reduce(%6688 init: %cst_1233) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6690 = stablehlo.convert %6689 : (tensor<f32>) -> tensor<bf16>
    %6691 = stablehlo.add %6686, %6690 : tensor<bf16>
    %6692 = stablehlo.multiply %5879, %5879 : tensor<2016x576xbf16>
    %6693 = stablehlo.convert %6692 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1234 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6694 = stablehlo.reduce(%6693 init: %cst_1234) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6695 = stablehlo.convert %6694 : (tensor<f32>) -> tensor<bf16>
    %6696 = stablehlo.add %6691, %6695 : tensor<bf16>
    %6697 = stablehlo.multiply %5880, %5880 : tensor<576xbf16>
    %6698 = stablehlo.convert %6697 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1235 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6699 = stablehlo.reduce(%6698 init: %cst_1235) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6700 = stablehlo.convert %6699 : (tensor<f32>) -> tensor<bf16>
    %6701 = stablehlo.add %6696, %6700 : tensor<bf16>
    %6702 = stablehlo.multiply %5881, %5881 : tensor<576xbf16>
    %6703 = stablehlo.convert %6702 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1236 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6704 = stablehlo.reduce(%6703 init: %cst_1236) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6705 = stablehlo.convert %6704 : (tensor<f32>) -> tensor<bf16>
    %6706 = stablehlo.add %6701, %6705 : tensor<bf16>
    %6707 = stablehlo.multiply %5882, %5882 : tensor<576x9x128xbf16>
    %6708 = stablehlo.convert %6707 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1237 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6709 = stablehlo.reduce(%6708 init: %cst_1237) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6710 = stablehlo.convert %6709 : (tensor<f32>) -> tensor<bf16>
    %6711 = stablehlo.add %6706, %6710 : tensor<bf16>
    %6712 = stablehlo.multiply %5883, %5883 : tensor<9x128x576xbf16>
    %6713 = stablehlo.convert %6712 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1238 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6714 = stablehlo.reduce(%6713 init: %cst_1238) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6715 = stablehlo.convert %6714 : (tensor<f32>) -> tensor<bf16>
    %6716 = stablehlo.add %6711, %6715 : tensor<bf16>
    %6717 = stablehlo.multiply %5884, %5884 : tensor<576x9x128xbf16>
    %6718 = stablehlo.convert %6717 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1239 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6719 = stablehlo.reduce(%6718 init: %cst_1239) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6720 = stablehlo.convert %6719 : (tensor<f32>) -> tensor<bf16>
    %6721 = stablehlo.add %6716, %6720 : tensor<bf16>
    %6722 = stablehlo.multiply %5885, %5885 : tensor<576x9x128xbf16>
    %6723 = stablehlo.convert %6722 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1240 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6724 = stablehlo.reduce(%6723 init: %cst_1240) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6725 = stablehlo.convert %6724 : (tensor<f32>) -> tensor<bf16>
    %6726 = stablehlo.add %6721, %6725 : tensor<bf16>
    %6727 = stablehlo.multiply %5886, %5886 : tensor<576x2016xbf16>
    %6728 = stablehlo.convert %6727 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1241 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6729 = stablehlo.reduce(%6728 init: %cst_1241) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6730 = stablehlo.convert %6729 : (tensor<f32>) -> tensor<bf16>
    %6731 = stablehlo.add %6726, %6730 : tensor<bf16>
    %6732 = stablehlo.multiply %5887, %5887 : tensor<576x2016xbf16>
    %6733 = stablehlo.convert %6732 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1242 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6734 = stablehlo.reduce(%6733 init: %cst_1242) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6735 = stablehlo.convert %6734 : (tensor<f32>) -> tensor<bf16>
    %6736 = stablehlo.add %6731, %6735 : tensor<bf16>
    %6737 = stablehlo.multiply %5888, %5888 : tensor<2016x576xbf16>
    %6738 = stablehlo.convert %6737 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1243 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6739 = stablehlo.reduce(%6738 init: %cst_1243) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6740 = stablehlo.convert %6739 : (tensor<f32>) -> tensor<bf16>
    %6741 = stablehlo.add %6736, %6740 : tensor<bf16>
    %6742 = stablehlo.multiply %5889, %5889 : tensor<576xbf16>
    %6743 = stablehlo.convert %6742 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1244 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6744 = stablehlo.reduce(%6743 init: %cst_1244) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6745 = stablehlo.convert %6744 : (tensor<f32>) -> tensor<bf16>
    %6746 = stablehlo.add %6741, %6745 : tensor<bf16>
    %6747 = stablehlo.multiply %5890, %5890 : tensor<576xbf16>
    %6748 = stablehlo.convert %6747 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1245 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6749 = stablehlo.reduce(%6748 init: %cst_1245) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6750 = stablehlo.convert %6749 : (tensor<f32>) -> tensor<bf16>
    %6751 = stablehlo.add %6746, %6750 : tensor<bf16>
    %6752 = stablehlo.multiply %5891, %5891 : tensor<576x9x128xbf16>
    %6753 = stablehlo.convert %6752 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1246 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6754 = stablehlo.reduce(%6753 init: %cst_1246) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6755 = stablehlo.convert %6754 : (tensor<f32>) -> tensor<bf16>
    %6756 = stablehlo.add %6751, %6755 : tensor<bf16>
    %6757 = stablehlo.multiply %5892, %5892 : tensor<9x128x576xbf16>
    %6758 = stablehlo.convert %6757 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1247 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6759 = stablehlo.reduce(%6758 init: %cst_1247) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6760 = stablehlo.convert %6759 : (tensor<f32>) -> tensor<bf16>
    %6761 = stablehlo.add %6756, %6760 : tensor<bf16>
    %6762 = stablehlo.multiply %5893, %5893 : tensor<576x9x128xbf16>
    %6763 = stablehlo.convert %6762 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1248 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6764 = stablehlo.reduce(%6763 init: %cst_1248) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6765 = stablehlo.convert %6764 : (tensor<f32>) -> tensor<bf16>
    %6766 = stablehlo.add %6761, %6765 : tensor<bf16>
    %6767 = stablehlo.multiply %5894, %5894 : tensor<576x9x128xbf16>
    %6768 = stablehlo.convert %6767 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1249 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6769 = stablehlo.reduce(%6768 init: %cst_1249) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6770 = stablehlo.convert %6769 : (tensor<f32>) -> tensor<bf16>
    %6771 = stablehlo.add %6766, %6770 : tensor<bf16>
    %6772 = stablehlo.multiply %5895, %5895 : tensor<576x2016xbf16>
    %6773 = stablehlo.convert %6772 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1250 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6774 = stablehlo.reduce(%6773 init: %cst_1250) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6775 = stablehlo.convert %6774 : (tensor<f32>) -> tensor<bf16>
    %6776 = stablehlo.add %6771, %6775 : tensor<bf16>
    %6777 = stablehlo.multiply %5896, %5896 : tensor<576x2016xbf16>
    %6778 = stablehlo.convert %6777 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1251 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6779 = stablehlo.reduce(%6778 init: %cst_1251) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6780 = stablehlo.convert %6779 : (tensor<f32>) -> tensor<bf16>
    %6781 = stablehlo.add %6776, %6780 : tensor<bf16>
    %6782 = stablehlo.multiply %5897, %5897 : tensor<2016x576xbf16>
    %6783 = stablehlo.convert %6782 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1252 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6784 = stablehlo.reduce(%6783 init: %cst_1252) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6785 = stablehlo.convert %6784 : (tensor<f32>) -> tensor<bf16>
    %6786 = stablehlo.add %6781, %6785 : tensor<bf16>
    %6787 = stablehlo.multiply %5898, %5898 : tensor<576xbf16>
    %6788 = stablehlo.convert %6787 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1253 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6789 = stablehlo.reduce(%6788 init: %cst_1253) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6790 = stablehlo.convert %6789 : (tensor<f32>) -> tensor<bf16>
    %6791 = stablehlo.add %6786, %6790 : tensor<bf16>
    %6792 = stablehlo.multiply %5899, %5899 : tensor<576xbf16>
    %6793 = stablehlo.convert %6792 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1254 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6794 = stablehlo.reduce(%6793 init: %cst_1254) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6795 = stablehlo.convert %6794 : (tensor<f32>) -> tensor<bf16>
    %6796 = stablehlo.add %6791, %6795 : tensor<bf16>
    %6797 = stablehlo.multiply %5900, %5900 : tensor<576x9x128xbf16>
    %6798 = stablehlo.convert %6797 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1255 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6799 = stablehlo.reduce(%6798 init: %cst_1255) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6800 = stablehlo.convert %6799 : (tensor<f32>) -> tensor<bf16>
    %6801 = stablehlo.add %6796, %6800 : tensor<bf16>
    %6802 = stablehlo.multiply %5901, %5901 : tensor<9x128x576xbf16>
    %6803 = stablehlo.convert %6802 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1256 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6804 = stablehlo.reduce(%6803 init: %cst_1256) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6805 = stablehlo.convert %6804 : (tensor<f32>) -> tensor<bf16>
    %6806 = stablehlo.add %6801, %6805 : tensor<bf16>
    %6807 = stablehlo.multiply %5902, %5902 : tensor<576x9x128xbf16>
    %6808 = stablehlo.convert %6807 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1257 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6809 = stablehlo.reduce(%6808 init: %cst_1257) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6810 = stablehlo.convert %6809 : (tensor<f32>) -> tensor<bf16>
    %6811 = stablehlo.add %6806, %6810 : tensor<bf16>
    %6812 = stablehlo.multiply %5903, %5903 : tensor<576x9x128xbf16>
    %6813 = stablehlo.convert %6812 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1258 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6814 = stablehlo.reduce(%6813 init: %cst_1258) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6815 = stablehlo.convert %6814 : (tensor<f32>) -> tensor<bf16>
    %6816 = stablehlo.add %6811, %6815 : tensor<bf16>
    %6817 = stablehlo.multiply %5904, %5904 : tensor<576x2016xbf16>
    %6818 = stablehlo.convert %6817 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1259 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6819 = stablehlo.reduce(%6818 init: %cst_1259) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6820 = stablehlo.convert %6819 : (tensor<f32>) -> tensor<bf16>
    %6821 = stablehlo.add %6816, %6820 : tensor<bf16>
    %6822 = stablehlo.multiply %5905, %5905 : tensor<576x2016xbf16>
    %6823 = stablehlo.convert %6822 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1260 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6824 = stablehlo.reduce(%6823 init: %cst_1260) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6825 = stablehlo.convert %6824 : (tensor<f32>) -> tensor<bf16>
    %6826 = stablehlo.add %6821, %6825 : tensor<bf16>
    %6827 = stablehlo.multiply %5906, %5906 : tensor<2016x576xbf16>
    %6828 = stablehlo.convert %6827 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1261 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6829 = stablehlo.reduce(%6828 init: %cst_1261) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6830 = stablehlo.convert %6829 : (tensor<f32>) -> tensor<bf16>
    %6831 = stablehlo.add %6826, %6830 : tensor<bf16>
    %6832 = stablehlo.multiply %5907, %5907 : tensor<576xbf16>
    %6833 = stablehlo.convert %6832 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1262 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6834 = stablehlo.reduce(%6833 init: %cst_1262) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6835 = stablehlo.convert %6834 : (tensor<f32>) -> tensor<bf16>
    %6836 = stablehlo.add %6831, %6835 : tensor<bf16>
    %6837 = stablehlo.multiply %5908, %5908 : tensor<576xbf16>
    %6838 = stablehlo.convert %6837 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1263 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6839 = stablehlo.reduce(%6838 init: %cst_1263) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6840 = stablehlo.convert %6839 : (tensor<f32>) -> tensor<bf16>
    %6841 = stablehlo.add %6836, %6840 : tensor<bf16>
    %6842 = stablehlo.multiply %5909, %5909 : tensor<576x9x128xbf16>
    %6843 = stablehlo.convert %6842 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1264 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6844 = stablehlo.reduce(%6843 init: %cst_1264) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6845 = stablehlo.convert %6844 : (tensor<f32>) -> tensor<bf16>
    %6846 = stablehlo.add %6841, %6845 : tensor<bf16>
    %6847 = stablehlo.multiply %5910, %5910 : tensor<9x128x576xbf16>
    %6848 = stablehlo.convert %6847 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1265 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6849 = stablehlo.reduce(%6848 init: %cst_1265) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6850 = stablehlo.convert %6849 : (tensor<f32>) -> tensor<bf16>
    %6851 = stablehlo.add %6846, %6850 : tensor<bf16>
    %6852 = stablehlo.multiply %5911, %5911 : tensor<576x9x128xbf16>
    %6853 = stablehlo.convert %6852 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1266 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6854 = stablehlo.reduce(%6853 init: %cst_1266) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6855 = stablehlo.convert %6854 : (tensor<f32>) -> tensor<bf16>
    %6856 = stablehlo.add %6851, %6855 : tensor<bf16>
    %6857 = stablehlo.multiply %5912, %5912 : tensor<576x9x128xbf16>
    %6858 = stablehlo.convert %6857 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1267 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6859 = stablehlo.reduce(%6858 init: %cst_1267) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6860 = stablehlo.convert %6859 : (tensor<f32>) -> tensor<bf16>
    %6861 = stablehlo.add %6856, %6860 : tensor<bf16>
    %6862 = stablehlo.multiply %5913, %5913 : tensor<576x2016xbf16>
    %6863 = stablehlo.convert %6862 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1268 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6864 = stablehlo.reduce(%6863 init: %cst_1268) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6865 = stablehlo.convert %6864 : (tensor<f32>) -> tensor<bf16>
    %6866 = stablehlo.add %6861, %6865 : tensor<bf16>
    %6867 = stablehlo.multiply %5914, %5914 : tensor<576x2016xbf16>
    %6868 = stablehlo.convert %6867 : (tensor<576x2016xbf16>) -> tensor<576x2016xf32>
    %cst_1269 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6869 = stablehlo.reduce(%6868 init: %cst_1269) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x2016xf32>, tensor<f32>) -> tensor<f32>
    %6870 = stablehlo.convert %6869 : (tensor<f32>) -> tensor<bf16>
    %6871 = stablehlo.add %6866, %6870 : tensor<bf16>
    %6872 = stablehlo.multiply %5915, %5915 : tensor<2016x576xbf16>
    %6873 = stablehlo.convert %6872 : (tensor<2016x576xbf16>) -> tensor<2016x576xf32>
    %cst_1270 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6874 = stablehlo.reduce(%6873 init: %cst_1270) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x576xf32>, tensor<f32>) -> tensor<f32>
    %6875 = stablehlo.convert %6874 : (tensor<f32>) -> tensor<bf16>
    %6876 = stablehlo.add %6871, %6875 : tensor<bf16>
    %6877 = stablehlo.multiply %5916, %5916 : tensor<576xbf16>
    %6878 = stablehlo.convert %6877 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1271 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6879 = stablehlo.reduce(%6878 init: %cst_1271) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6880 = stablehlo.convert %6879 : (tensor<f32>) -> tensor<bf16>
    %6881 = stablehlo.add %6876, %6880 : tensor<bf16>
    %6882 = stablehlo.multiply %5917, %5917 : tensor<576xbf16>
    %6883 = stablehlo.convert %6882 : (tensor<576xbf16>) -> tensor<576xf32>
    %cst_1272 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6884 = stablehlo.reduce(%6883 init: %cst_1272) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %6885 = stablehlo.convert %6884 : (tensor<f32>) -> tensor<bf16>
    %6886 = stablehlo.add %6881, %6885 : tensor<bf16>
    %6887 = stablehlo.multiply %5918, %5918 : tensor<576x9x128xbf16>
    %6888 = stablehlo.convert %6887 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1273 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6889 = stablehlo.reduce(%6888 init: %cst_1273) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6890 = stablehlo.convert %6889 : (tensor<f32>) -> tensor<bf16>
    %6891 = stablehlo.add %6886, %6890 : tensor<bf16>
    %6892 = stablehlo.multiply %5919, %5919 : tensor<9x128x576xbf16>
    %6893 = stablehlo.convert %6892 : (tensor<9x128x576xbf16>) -> tensor<9x128x576xf32>
    %cst_1274 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6894 = stablehlo.reduce(%6893 init: %cst_1274) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x576xf32>, tensor<f32>) -> tensor<f32>
    %6895 = stablehlo.convert %6894 : (tensor<f32>) -> tensor<bf16>
    %6896 = stablehlo.add %6891, %6895 : tensor<bf16>
    %6897 = stablehlo.multiply %5920, %5920 : tensor<576x9x128xbf16>
    %6898 = stablehlo.convert %6897 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1275 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6899 = stablehlo.reduce(%6898 init: %cst_1275) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6900 = stablehlo.convert %6899 : (tensor<f32>) -> tensor<bf16>
    %6901 = stablehlo.add %6896, %6900 : tensor<bf16>
    %6902 = stablehlo.multiply %5921, %5921 : tensor<576x9x128xbf16>
    %6903 = stablehlo.convert %6902 : (tensor<576x9x128xbf16>) -> tensor<576x9x128xf32>
    %cst_1276 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6904 = stablehlo.reduce(%6903 init: %cst_1276) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<576x9x128xf32>, tensor<f32>) -> tensor<f32>
    %6905 = stablehlo.convert %6904 : (tensor<f32>) -> tensor<bf16>
    %6906 = stablehlo.add %6901, %6905 : tensor<bf16>
    %6907 = stablehlo.multiply %5922, %5922 : tensor<576x128256xbf16>
    %6908 = stablehlo.convert %6907 : (tensor<576x128256xbf16>) -> tensor<576x128256xf32>
    %cst_1277 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6909 = stablehlo.reduce(%6908 init: %cst_1277) applies stablehlo.add across dimensions = [0, 1] : (tensor<576x128256xf32>, tensor<f32>) -> tensor<f32>
    %6910 = stablehlo.convert %6909 : (tensor<f32>) -> tensor<bf16>
    %6911 = stablehlo.add %6906, %6910 : tensor<bf16>
    %6912 = stablehlo.multiply %5923, %5923 : tensor<128256x576xbf16>
    %6913 = stablehlo.convert %6912 : (tensor<128256x576xbf16>) -> tensor<128256x576xf32>
    %cst_1278 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6914 = stablehlo.reduce(%6913 init: %cst_1278) applies stablehlo.add across dimensions = [0, 1] : (tensor<128256x576xf32>, tensor<f32>) -> tensor<f32>
    %6915 = stablehlo.convert %6914 : (tensor<f32>) -> tensor<bf16>
    %6916 = stablehlo.add %6911, %6915 : tensor<bf16>
    %6917 = stablehlo.sqrt %6916 : tensor<bf16>
    %cst_1279 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1280 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    return %5924, %5858, %5859, %5860, %5861, %5862, %5863, %5864, %5865, %5866, %5867, %5868, %5869, %5870, %5871, %5872, %5873, %5874, %5875, %5876, %5877, %5878, %5879, %5880, %5881, %5882, %5883, %5884, %5885, %5886, %5887, %5888, %5889, %5890, %5891, %5892, %5893, %5894, %5895, %5896, %5897, %5898, %5899, %5900, %5901, %5902, %5903, %5904, %5905, %5906, %5907, %5908, %5909, %5910, %5911, %5912, %5913, %5914, %5915, %5916, %5917, %5918, %5919, %5920, %5921, %5922, %5923, %4639, %3915, %3920, %3925, %3930, %3935, %3940, %3945, %3950, %3955, %3960, %3965, %3970, %3975, %3980, %3985, %3990, %3995, %4000, %4005, %4010, %4015, %4020, %4025, %4030, %4035, %4040, %4045, %4050, %4055, %4060, %4065, %4070, %4075, %4080, %4085, %4090, %4095, %4100, %4105, %4110, %4115, %4120, %4125, %4130, %4135, %4140, %4145, %4150, %4155, %4160, %4165, %4170, %4175, %4180, %4185, %4190, %4195, %4200, %4205, %4210, %4215, %4220, %4225, %4230, %4235, %4240, %4246, %4252, %4258, %4264, %4270, %4276, %4282, %4288, %4294, %4300, %4306, %4312, %4318, %4324, %4330, %4336, %4342, %4348, %4354, %4360, %4366, %4372, %4378, %4384, %4390, %4396, %4402, %4408, %4414, %4420, %4426, %4432, %4438, %4444, %4450, %4456, %4462, %4468, %4474, %4480, %4486, %4492, %4498, %4504, %4510, %4516, %4522, %4528, %4534, %4540, %4546, %4552, %4558, %4564, %4570, %4576, %4582, %4588, %4594, %4600, %4606, %4612, %4618, %4624, %4630, %4636, %5857, %6255, %1207, %cst_1279, %cst_1280, %6917, %6586, %1203 : tensor<i32>, tensor<576xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x128256xbf16>, tensor<128256x576xbf16>, tensor<i32>, tensor<576xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x128256xbf16>, tensor<128256x576xbf16>, tensor<576xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x2016xbf16>, tensor<576x2016xbf16>, tensor<2016x576xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<576x9x128xbf16>, tensor<9x128x576xbf16>, tensor<576x9x128xbf16>, tensor<576x9x128xbf16>, tensor<576x128256xbf16>, tensor<128256x576xbf16>, tensor<i32>, tensor<bf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<bf16>, tensor<bf16>, tensor<i32>
  }
  func.func private @_where(%arg0: tensor<8x1x1x4096x4096xi1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<8x1x1x4096x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %2 = stablehlo.select %arg0, %0, %1 : tensor<8x1x1x4096x4096xi1>, tensor<8x1x1x4096x4096xf32>
    return %2 : tensor<8x1x1x4096x4096xf32>
  }
  func.func private @_where_34(%arg0: tensor<8x1x1x4096x4096xi1>, %arg1: tensor<8x9x1x4096x4096xbf16>, %arg2: tensor<f32>) -> tensor<8x9x1x4096x4096xbf16> {
    %0 = stablehlo.convert %arg2 : (tensor<f32>) -> tensor<bf16>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3, 4] : (tensor<8x1x1x4096x4096xi1>) -> tensor<8x9x1x4096x4096xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<bf16>) -> tensor<8x9x1x4096x4096xbf16>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<8x9x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>
    return %3 : tensor<8x9x1x4096x4096xbf16>
  }
  func.func private @silu(%arg0: tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>) {
    %0 = stablehlo.negate %arg0 : tensor<8x4096x2016xbf16>
    %1 = stablehlo.exponential %0 : tensor<8x4096x2016xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<8x4096x2016xbf16>
    %3 = stablehlo.add %2, %1 : tensor<8x4096x2016xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<8x4096x2016xbf16>
    %5 = stablehlo.divide %4, %3 : tensor<8x4096x2016xbf16>
    %6 = stablehlo.multiply %arg0, %5 : tensor<8x4096x2016xbf16>
    return %6, %arg0 : tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>
  }
  func.func private @_one_hot(%arg0: tensor<8x4096xi32>) -> tensor<8x4096x128256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<8x4096xi32>) -> tensor<8x4096x1xi32>
    %1 = stablehlo.iota dim = 2 : tensor<1x1x128256xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<8x4096x1xi32>) -> tensor<8x4096x128256xi32>
    %3 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<1x1x128256xi32>) -> tensor<8x4096x128256xi32>
    %4 = stablehlo.compare  EQ, %2, %3,  SIGNED : (tensor<8x4096x128256xi32>, tensor<8x4096x128256xi32>) -> tensor<8x4096x128256xi1>
    %5 = stablehlo.convert %4 : (tensor<8x4096x128256xi1>) -> tensor<8x4096x128256xf32>
    return %5 : tensor<8x4096x128256xf32>
  }
  func.func private @_where_115(%arg0: tensor<8x1x1x4096x4096xi1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<8x1x1x4096x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<8x1x1x4096x4096xf32>
    %2 = stablehlo.select %arg0, %0, %1 : tensor<8x1x1x4096x4096xi1>, tensor<8x1x1x4096x4096xf32>
    return %2 : tensor<8x1x1x4096x4096xf32>
  }
  func.func private @_where_116(%arg0: tensor<8x1x1x4096x4096xi1>, %arg1: tensor<8x9x1x4096x4096xbf16>, %arg2: tensor<f32>) -> (tensor<8x9x1x4096x4096xbf16>, tensor<8x9x1x4096x4096xi1>) {
    %0 = stablehlo.convert %arg2 : (tensor<f32>) -> tensor<bf16>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3, 4] : (tensor<8x1x1x4096x4096xi1>) -> tensor<8x9x1x4096x4096xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<bf16>) -> tensor<8x9x1x4096x4096xbf16>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<8x9x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>
    return %3, %1 : tensor<8x9x1x4096x4096xbf16>, tensor<8x9x1x4096x4096xi1>
  }
  func.func private @silu_124(%arg0: tensor<8x4096x2016xbf16>) -> (tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>) {
    %0 = stablehlo.negate %arg0 : tensor<8x4096x2016xbf16>
    %1 = stablehlo.exponential %0 : tensor<8x4096x2016xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<8x4096x2016xbf16>
    %3 = stablehlo.add %2, %1 : tensor<8x4096x2016xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<8x4096x2016xbf16>
    %5 = stablehlo.divide %4, %3 : tensor<8x4096x2016xbf16>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<bf16>) -> tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<bf16>) -> tensor<8x4096x2016xbf16>
    %8 = stablehlo.subtract %7, %5 : tensor<8x4096x2016xbf16>
    %9 = stablehlo.multiply %5, %8 : tensor<8x4096x2016xbf16>
    %10 = stablehlo.multiply %arg0, %5 : tensor<8x4096x2016xbf16>
    return %10, %5, %9 : tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>, tensor<8x4096x2016xbf16>
  }
  func.func private @silu_133(%arg0: tensor<8x4096x2016xbf16>, %arg1: tensor<8x4096x2016xbf16>, %arg2: tensor<8x4096x2016xbf16>, %arg3: tensor<8x4096x2016xbf16>) -> tensor<8x4096x2016xbf16> {
    %0 = stablehlo.multiply %arg1, %arg3 : tensor<8x4096x2016xbf16>
    %1 = stablehlo.multiply %arg3, %arg0 : tensor<8x4096x2016xbf16>
    %2 = stablehlo.multiply %0, %arg2 : tensor<8x4096x2016xbf16>
    %3 = stablehlo.add %1, %2 : tensor<8x4096x2016xbf16>
    return %3 : tensor<8x4096x2016xbf16>
  }
  func.func private @_where_163(%arg0: tensor<8x9x1x4096x4096xi1>, %arg1: tensor<8x9x1x4096x4096xbf16>) -> tensor<8x9x1x4096x4096xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<8x9x1x4096x4096xbf16>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<8x9x1x4096x4096xi1>, tensor<8x9x1x4096x4096xbf16>
    return %1 : tensor<8x9x1x4096x4096xbf16>
  }
  func.func private @_where_260(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @clip(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<i32>
    %2 = stablehlo.convert %arg2 : tensor<i32>
    %3 = stablehlo.minimum %2, %1 : tensor<i32>
    return %3 : tensor<i32>
  }
  func.func private @_where_293(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
}
