module @jit_train_step attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [{alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 0 : i64}, output_index = array<i64: 0>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 1 : i64}, output_index = array<i64: 1>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 2 : i64}, output_index = array<i64: 2>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 3 : i64}, output_index = array<i64: 3>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 4 : i64}, output_index = array<i64: 4>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 5 : i64}, output_index = array<i64: 5>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 6 : i64}, output_index = array<i64: 6>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 7 : i64}, output_index = array<i64: 7>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 8 : i64}, output_index = array<i64: 8>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 9 : i64}, output_index = array<i64: 9>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 10 : i64}, output_index = array<i64: 10>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 11 : i64}, output_index = array<i64: 11>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 12 : i64}, output_index = array<i64: 12>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 13 : i64}, output_index = array<i64: 13>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 14 : i64}, output_index = array<i64: 14>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 15 : i64}, output_index = array<i64: 15>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 16 : i64}, output_index = array<i64: 16>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 17 : i64}, output_index = array<i64: 17>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 18 : i64}, output_index = array<i64: 18>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 19 : i64}, output_index = array<i64: 19>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 20 : i64}, output_index = array<i64: 20>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 21 : i64}, output_index = array<i64: 21>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 22 : i64}, output_index = array<i64: 22>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 23 : i64}, output_index = array<i64: 23>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 24 : i64}, output_index = array<i64: 24>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 25 : i64}, output_index = array<i64: 25>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 26 : i64}, output_index = array<i64: 26>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 27 : i64}, output_index = array<i64: 27>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 28 : i64}, output_index = array<i64: 28>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 29 : i64}, output_index = array<i64: 29>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 30 : i64}, output_index = array<i64: 30>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 31 : i64}, output_index = array<i64: 31>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 32 : i64}, output_index = array<i64: 32>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 33 : i64}, output_index = array<i64: 33>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 34 : i64}, output_index = array<i64: 34>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 35 : i64}, output_index = array<i64: 35>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 36 : i64}, output_index = array<i64: 36>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 37 : i64}, output_index = array<i64: 37>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 38 : i64}, output_index = array<i64: 38>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 39 : i64}, output_index = array<i64: 39>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 40 : i64}, output_index = array<i64: 40>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 41 : i64}, output_index = array<i64: 41>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 42 : i64}, output_index = array<i64: 42>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 43 : i64}, output_index = array<i64: 43>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 44 : i64}, output_index = array<i64: 44>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 45 : i64}, output_index = array<i64: 45>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 46 : i64}, output_index = array<i64: 46>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 47 : i64}, output_index = array<i64: 47>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 48 : i64}, output_index = array<i64: 48>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 49 : i64}, output_index = array<i64: 49>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 50 : i64}, output_index = array<i64: 50>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 51 : i64}, output_index = array<i64: 51>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 52 : i64}, output_index = array<i64: 52>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 53 : i64}, output_index = array<i64: 53>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 54 : i64}, output_index = array<i64: 54>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 55 : i64}, output_index = array<i64: 55>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 56 : i64}, output_index = array<i64: 56>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 57 : i64}, output_index = array<i64: 57>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 58 : i64}, output_index = array<i64: 58>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 59 : i64}, output_index = array<i64: 59>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 60 : i64}, output_index = array<i64: 60>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 61 : i64}, output_index = array<i64: 61>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 62 : i64}, output_index = array<i64: 62>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 63 : i64}, output_index = array<i64: 63>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 64 : i64}, output_index = array<i64: 64>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 65 : i64}, output_index = array<i64: 65>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 66 : i64}, output_index = array<i64: 66>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 67 : i64}, output_index = array<i64: 67>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 68 : i64}, output_index = array<i64: 68>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 69 : i64}, output_index = array<i64: 69>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 70 : i64}, output_index = array<i64: 70>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 71 : i64}, output_index = array<i64: 71>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 72 : i64}, output_index = array<i64: 72>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 73 : i64}, output_index = array<i64: 73>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 74 : i64}, output_index = array<i64: 74>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 75 : i64}, output_index = array<i64: 75>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 76 : i64}, output_index = array<i64: 76>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 77 : i64}, output_index = array<i64: 77>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 78 : i64}, output_index = array<i64: 78>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 79 : i64}, output_index = array<i64: 79>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 80 : i64}, output_index = array<i64: 80>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 81 : i64}, output_index = array<i64: 81>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 82 : i64}, output_index = array<i64: 82>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 83 : i64}, output_index = array<i64: 83>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 84 : i64}, output_index = array<i64: 84>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 85 : i64}, output_index = array<i64: 85>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 86 : i64}, output_index = array<i64: 86>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 87 : i64}, output_index = array<i64: 87>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 88 : i64}, output_index = array<i64: 88>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 89 : i64}, output_index = array<i64: 89>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 90 : i64}, output_index = array<i64: 90>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 91 : i64}, output_index = array<i64: 91>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 92 : i64}, output_index = array<i64: 92>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 93 : i64}, output_index = array<i64: 93>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 94 : i64}, output_index = array<i64: 94>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 95 : i64}, output_index = array<i64: 95>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 96 : i64}, output_index = array<i64: 96>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 97 : i64}, output_index = array<i64: 97>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 98 : i64}, output_index = array<i64: 98>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 99 : i64}, output_index = array<i64: 99>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 100 : i64}, output_index = array<i64: 100>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 101 : i64}, output_index = array<i64: 101>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 102 : i64}, output_index = array<i64: 102>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 103 : i64}, output_index = array<i64: 103>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 104 : i64}, output_index = array<i64: 104>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 105 : i64}, output_index = array<i64: 105>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 106 : i64}, output_index = array<i64: 106>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 107 : i64}, output_index = array<i64: 107>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 108 : i64}, output_index = array<i64: 108>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 109 : i64}, output_index = array<i64: 109>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 110 : i64}, output_index = array<i64: 110>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 111 : i64}, output_index = array<i64: 111>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 112 : i64}, output_index = array<i64: 112>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 113 : i64}, output_index = array<i64: 113>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 114 : i64}, output_index = array<i64: 114>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 115 : i64}, output_index = array<i64: 115>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 116 : i64}, output_index = array<i64: 116>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 117 : i64}, output_index = array<i64: 117>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 118 : i64}, output_index = array<i64: 118>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 119 : i64}, output_index = array<i64: 119>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 120 : i64}, output_index = array<i64: 120>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 121 : i64}, output_index = array<i64: 121>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 122 : i64}, output_index = array<i64: 122>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 123 : i64}, output_index = array<i64: 123>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 124 : i64}, output_index = array<i64: 124>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 125 : i64}, output_index = array<i64: 125>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 126 : i64}, output_index = array<i64: 126>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 127 : i64}, output_index = array<i64: 127>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 128 : i64}, output_index = array<i64: 128>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 129 : i64}, output_index = array<i64: 129>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 130 : i64}, output_index = array<i64: 130>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 131 : i64}, output_index = array<i64: 131>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 132 : i64}, output_index = array<i64: 132>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 133 : i64}, output_index = array<i64: 133>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 134 : i64}, output_index = array<i64: 134>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 135 : i64}, output_index = array<i64: 135>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 136 : i64}, output_index = array<i64: 136>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 137 : i64}, output_index = array<i64: 137>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 138 : i64}, output_index = array<i64: 138>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 139 : i64}, output_index = array<i64: 139>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 140 : i64}, output_index = array<i64: 140>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 141 : i64}, output_index = array<i64: 141>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 142 : i64}, output_index = array<i64: 142>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 143 : i64}, output_index = array<i64: 143>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 144 : i64}, output_index = array<i64: 144>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 145 : i64}, output_index = array<i64: 145>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 146 : i64}, output_index = array<i64: 146>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 147 : i64}, output_index = array<i64: 147>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 148 : i64}, output_index = array<i64: 148>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 149 : i64}, output_index = array<i64: 149>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 150 : i64}, output_index = array<i64: 150>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 151 : i64}, output_index = array<i64: 151>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 152 : i64}, output_index = array<i64: 152>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 153 : i64}, output_index = array<i64: 153>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 154 : i64}, output_index = array<i64: 154>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 155 : i64}, output_index = array<i64: 155>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 156 : i64}, output_index = array<i64: 156>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 157 : i64}, output_index = array<i64: 157>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 158 : i64}, output_index = array<i64: 158>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 159 : i64}, output_index = array<i64: 159>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 160 : i64}, output_index = array<i64: 160>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 161 : i64}, output_index = array<i64: 161>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 162 : i64}, output_index = array<i64: 162>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 163 : i64}, output_index = array<i64: 163>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 164 : i64}, output_index = array<i64: 164>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 165 : i64}, output_index = array<i64: 165>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 166 : i64}, output_index = array<i64: 166>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 167 : i64}, output_index = array<i64: 167>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 168 : i64}, output_index = array<i64: 168>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 169 : i64}, output_index = array<i64: 169>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 170 : i64}, output_index = array<i64: 170>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 171 : i64}, output_index = array<i64: 171>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 172 : i64}, output_index = array<i64: 172>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 173 : i64}, output_index = array<i64: 173>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 174 : i64}, output_index = array<i64: 174>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 175 : i64}, output_index = array<i64: 175>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 176 : i64}, output_index = array<i64: 176>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 177 : i64}, output_index = array<i64: 177>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 178 : i64}, output_index = array<i64: 178>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 179 : i64}, output_index = array<i64: 179>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 180 : i64}, output_index = array<i64: 180>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 181 : i64}, output_index = array<i64: 181>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 182 : i64}, output_index = array<i64: 182>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 183 : i64}, output_index = array<i64: 183>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 184 : i64}, output_index = array<i64: 184>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 185 : i64}, output_index = array<i64: 185>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 186 : i64}, output_index = array<i64: 186>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 187 : i64}, output_index = array<i64: 187>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 188 : i64}, output_index = array<i64: 188>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 189 : i64}, output_index = array<i64: 189>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 190 : i64}, output_index = array<i64: 190>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 191 : i64}, output_index = array<i64: 191>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 192 : i64}, output_index = array<i64: 192>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 193 : i64}, output_index = array<i64: 193>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 194 : i64}, output_index = array<i64: 194>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 195 : i64}, output_index = array<i64: 195>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 196 : i64}, output_index = array<i64: 196>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 197 : i64}, output_index = array<i64: 197>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 198 : i64}, output_index = array<i64: 198>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 199 : i64}, output_index = array<i64: 199>}, {alias = {kind = "may_alias", parameter_index = array<i64>, parameter_number = 200 : i64}, output_index = array<i64: 200>}], mhlo.is_dynamic = false, mhlo.num_partitions = 4 : i32, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @region_289.302(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_290.303(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_291.304(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_292.305(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_293.306(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_294.307(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_295.308(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_296.309(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_297.310(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_298.311(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_299.312(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_300.313(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_301.314(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_302.315(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_303.316(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_304.317(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_305.318(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_306.319(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_307.320(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_308.321(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_309.322(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_310.323(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_311.324(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_312.325(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_313.326(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_314.327(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_315.328(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_316.329(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_317.330(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_318.331(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_319.332(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_320.333(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_321.334(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_322.335(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_323.336(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_324.337(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_325.338(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_326.339(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_327.340(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_328.341(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_329.342(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_330.343(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_331.344(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_332.345(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_333.346(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_334.347(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_335.348(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_336.349(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_337.350(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_338.351(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_339.352(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_340.353(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_341.354(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_342.355(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_343.356(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_344.357(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_345.358(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_346.359(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_347.360(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_348.361(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_349.362(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_350.363(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_351.364(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_352.365(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_353.366(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_354.367(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_2.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_3.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_4.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.9(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_6.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_8.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_9.13(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_10.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_11.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_12.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_13.17(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_14.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_15.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_16.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_17.21(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_18.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_19.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_20.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_21.25(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_22.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_23.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_24.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_25.29(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_26.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_27.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_28.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_33.38(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @region_33.38.clone(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @region_29.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_30.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_34.39(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_34.39.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_157.167(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_36.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_42.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_45.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_39.46(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_41.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_47.56(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_49.58(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_40.47(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_38.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_53.63(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_59.69(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_62.72(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_56.66(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_58.68(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_64.74(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_66.76(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_57.67(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_55.65(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_70.80(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_76.86(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_79.89(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_73.83(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_75.85(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_81.91(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_83.93(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_74.84(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_72.82(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_87.97(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_93.103(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_96.106(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_90.100(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_92.102(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_98.108(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_100.110(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_91.101(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_89.99(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_104.114(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_110.120(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_113.123(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_107.117(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_109.119(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_115.125(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_117.127(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_108.118(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_106.116(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_121.131(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_127.137(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_130.140(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_124.134(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_126.136(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_132.142(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_134.144(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_125.135(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_123.133(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_138.148(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_144.154(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_158.168(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_158.168.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.1.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_159.169(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_159.169.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.2.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_160.170(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_160.170.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_145.155(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_145.155.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_161.171(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_140.150(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_147.157(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_141.151(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_143.153(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_149.159(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_151.161(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_142.152(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_153.163(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_153.163.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_162.172(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.3.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_163.173(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_163.173.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.4.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_164.174(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_164.174.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.5.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_165.175(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_165.175.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.6.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_166.176(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_166.176.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.7.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_167.177(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_167.177.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.8.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_168.178(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_168.178.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.9.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_169.179(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_169.179.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_128.138(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_128.138.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_170.180(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_136.146(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_136.146.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_171.181(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.10.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_172.182(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_172.182.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.11.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_173.183(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_173.183.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.12.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_174.184(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_174.184.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.13.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_175.185(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_175.185.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.14.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_176.186(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_176.186.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.15.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_177.187(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_177.187.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.16.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_178.188(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_178.188.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_111.121(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_111.121.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_179.189(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_119.129(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_119.129.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_180.190(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.17.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_181.191(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_181.191.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.18.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_182.192(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_182.192.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.19.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_183.193(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_183.193.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.20.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_184.194(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_184.194.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.21.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_185.195(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_185.195.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.22.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_186.196(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_186.196.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.23.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_187.197(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_187.197.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_94.104(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_94.104.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_188.198(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_102.112(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_102.112.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_189.199(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.24.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_190.200(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_190.200.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.25.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_191.201(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_191.201.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.26.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_192.202(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_192.202.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.27.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_193.203(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_193.203.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.28.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_194.204(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_194.204.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.29.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_195.205(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_195.205.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.30.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_196.206(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_196.206.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_77.87(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_77.87.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_197.207(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_85.95(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_85.95.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_198.208(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.31.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_199.209(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_199.209.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.32.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_200.210(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_200.210.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.33.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_201.211(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_201.211.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.34.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_202.212(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_202.212.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.35.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_203.213(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_203.213.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.36.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_204.214(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_204.214.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.37.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_205.215(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_205.215.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_60.70(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_60.70.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_206.216(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_68.78(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_68.78.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_207.217(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.38.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_208.218(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_208.218.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.39.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_209.219(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_209.219.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.40.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_210.220(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_210.220.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.41.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_211.221(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_211.221.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.42.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_212.222(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_212.222.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.43.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_213.223(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_213.223.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.44.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_214.224(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_214.224.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_43.52(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_43.52.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_215.225(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_51.61(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_51.61.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_216.226(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.45.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_217.227(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_217.227.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.46.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_218.228(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_218.228.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.47.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_219.229(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_219.229.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.48.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_220.230(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_220.230.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.49.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_221.231(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_221.231.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_155.165(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @add.50.clone(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func private @region_222.232(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_222.232.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_223.236(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_224.237(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_224.237.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_225.238(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_225.238.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_226.239(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_226.239.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_227.240(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_228.241(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_229.242(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_229.242.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_230.243(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_230.243.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_231.244(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_231.244.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_232.245(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_232.245.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_233.246(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_233.246.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_234.247(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_234.247.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_235.248(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_235.248.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_236.249(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_237.250(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_238.251(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_238.251.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_239.252(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_239.252.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_240.253(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_240.253.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_241.254(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_241.254.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_242.255(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_242.255.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_243.256(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_243.256.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_244.257(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_244.257.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_245.258(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_246.259(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_247.260(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_247.260.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_248.261(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_248.261.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_249.262(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_249.262.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_250.263(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_250.263.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_251.264(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_251.264.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_252.265(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_252.265.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_253.266(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_253.266.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_254.267(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_255.268(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_256.269(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_256.269.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_257.270(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_257.270.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_258.271(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_258.271.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_259.272(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_259.272.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_260.273(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_260.273.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_261.274(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_261.274.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_262.275(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_262.275.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_263.276(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_264.277(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_265.278(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_265.278.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_266.279(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_266.279.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_267.280(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_267.280.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_268.281(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_268.281.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_269.282(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_269.282.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_270.283(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_270.283.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_271.284(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_271.284.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_272.285(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_273.286(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_274.287(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_274.287.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_275.288(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_275.288.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_276.289(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_276.289.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_277.290(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_277.290.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_278.291(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_278.291.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_279.292(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_279.292.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_280.293(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_280.293.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_281.294(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_282.295(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_283.296(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_283.296.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_284.297(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_284.297.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_285.298(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_285.298.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_286.299(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_286.299.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_287.300(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_287.300.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_288.301(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_288.301.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_31.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_32.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_32.37.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_355.368(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_356.369(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_356.369.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_357.370(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_357.370.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_358.371(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_358.371.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_359.372(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_360.373(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_361.374(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_361.374.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_362.375(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_362.375.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_363.376(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_363.376.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_364.377(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_364.377.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_365.378(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_365.378.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_366.379(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_366.379.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_367.380(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_367.380.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_368.381(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_369.382(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_370.383(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_370.383.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_371.384(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_371.384.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_372.385(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_372.385.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_373.386(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_373.386.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_374.387(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_374.387.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_375.388(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_375.388.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_376.389(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_376.389.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_377.390(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_378.391(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_379.392(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_379.392.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_380.393(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_380.393.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_381.394(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_381.394.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_382.395(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_382.395.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_383.396(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_383.396.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_384.397(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_384.397.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_385.398(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_385.398.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_386.399(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_387.400(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_388.401(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_388.401.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_389.402(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_389.402.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_390.403(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_390.403.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_391.404(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_391.404.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_392.405(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_392.405.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_393.406(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_393.406.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_394.407(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_394.407.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_395.408(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_396.409(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_397.410(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_397.410.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_398.411(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_398.411.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_399.412(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_399.412.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_400.413(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_400.413.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_401.414(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_401.414.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_402.415(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_402.415.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_403.416(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_403.416.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_404.417(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_405.418(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_406.419(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_406.419.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_407.420(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_407.420.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_408.421(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_408.421.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_409.422(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_409.422.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_410.423(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_410.423.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_411.424(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_411.424.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_412.425(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_412.425.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_413.426(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_414.427(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_415.428(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_415.428.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_416.429(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_416.429.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_417.430(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_417.430.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_418.431(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_418.431.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_419.432(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_419.432.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_420.433(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_420.433.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_290.303.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_291.304.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_292.305.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_295.308.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_296.309.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_297.310.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_298.311.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_299.312.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_300.313.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_301.314.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_304.317.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_305.318.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_306.319.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_307.320.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_308.321.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_309.322.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_310.323.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_313.326.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_314.327.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_315.328.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_316.329.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_317.330.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_318.331.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_319.332.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_322.335.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_323.336.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_324.337.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_325.338.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_326.339.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_327.340.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_328.341.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_331.344.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_332.345.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_333.346.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_334.347.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_335.348.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_336.349.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_337.350.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_340.353.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_341.354.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_342.355.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_343.356.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_344.357.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_345.358.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_346.359.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_349.362.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_350.363.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_351.364.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_352.365.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_353.366.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_354.367.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func @main(%arg0: tensor<i32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg2: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg3: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg4: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg5: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg6: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg7: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg8: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg9: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg10: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg11: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg12: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg13: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg14: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg15: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg16: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg17: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg18: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg19: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg20: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg21: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg22: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg23: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg24: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg25: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg26: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg27: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg28: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg29: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg30: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg31: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg32: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg33: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg34: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg35: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg36: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg37: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg38: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg39: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg40: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg41: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg42: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg43: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg44: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg45: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg46: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg47: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg48: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg49: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg50: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg51: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg52: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg53: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg54: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg55: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg56: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg57: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg58: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg59: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg60: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg61: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg62: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg63: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg64: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg65: tensor<144x128256xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg66: tensor<128256x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg67: tensor<i32> {mhlo.sharding = "{replicated}"}, %arg68: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg69: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg70: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg71: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg72: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg73: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg74: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg75: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg76: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg77: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg78: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg79: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg80: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg81: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg82: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg83: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg84: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg85: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg86: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg87: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg88: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg89: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg90: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg91: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg92: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg93: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg94: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg95: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg96: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg97: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg98: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg99: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg100: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg101: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg102: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg103: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg104: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg105: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg106: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg107: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg108: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg109: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg110: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg111: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg112: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg113: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg114: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg115: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg116: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg117: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg118: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg119: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg120: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg121: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg122: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg123: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg124: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg125: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg126: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg127: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg128: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg129: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg130: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg131: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg132: tensor<144x128256xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg133: tensor<128256x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg134: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg135: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg136: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg137: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg138: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg139: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg140: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg141: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg142: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg143: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg144: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg145: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg146: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg147: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg148: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg149: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg150: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg151: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg152: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg153: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg154: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg155: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg156: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg157: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg158: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg159: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg160: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg161: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg162: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg163: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg164: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg165: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg166: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg167: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg168: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg169: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg170: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg171: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg172: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg173: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg174: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg175: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg176: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg177: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg178: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg179: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg180: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg181: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg182: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg183: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg184: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg185: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg186: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg187: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg188: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg189: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg190: tensor<144x2016xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg191: tensor<2016x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg192: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg193: tensor<576xbf16> {mhlo.sharding = "{replicated}"}, %arg194: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg195: tensor<9x128x144xbf16> {mhlo.sharding = "{devices=[1,1,4]<=[4]}"}, %arg196: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg197: tensor<144x9x128xbf16> {mhlo.sharding = "{devices=[4,1,1]<=[4]}"}, %arg198: tensor<144x128256xbf16> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg199: tensor<128256x144xbf16> {mhlo.sharding = "{devices=[1,4]<=[4]}"}, %arg200: tensor<i32> {mhlo.sharding = "{replicated}"}, %arg201: tensor<1x4096xi32> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg202: tensor<1x4096xi32> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg203: tensor<1x4096xi32> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg204: tensor<1x4096xi32> {mhlo.sharding = "{devices=[4,1]<=[4]}"}, %arg205: tensor<1x4096xi32> {mhlo.sharding = "{devices=[4,1]<=[4]}"}) -> (tensor<i32>, tensor<576xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x128256xbf16>, tensor<128256x144xbf16>, tensor<i32>, tensor<576xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x128256xbf16>, tensor<128256x144xbf16>, tensor<576xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x128256xbf16>, tensor<128256x144xbf16>, tensor<i32>, tensor<bf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<bf16>, tensor<bf16>, tensor<i32>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg0, %c : tensor<i32>
    %c_0 = stablehlo.constant dense<3> : tensor<i32>
    %1 = stablehlo.compare  LT, %arg200, %c_0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.maximum %c_1, %arg200 : tensor<i32>
    %3 = stablehlo.minimum %c_0, %2 : tensor<i32>
    %4 = stablehlo.convert %3 : (tensor<i32>) -> tensor<f32>
    %cst_2 = stablehlo.constant dense<0.333333343> : tensor<f32>
    %5 = stablehlo.multiply %4, %cst_2 : tensor<f32>
    %6 = stablehlo.subtract %cst, %5 : tensor<f32>
    %cst_3 = stablehlo.constant dense<-3.000000e-05> : tensor<f32>
    %7 = stablehlo.multiply %6, %cst_3 : tensor<f32>
    %cst_4 = stablehlo.constant dense<3.000000e-05> : tensor<f32>
    %8 = stablehlo.add %7, %cst_4 : tensor<f32>
    %c_5 = stablehlo.constant dense<-3> : tensor<i32>
    %9 = stablehlo.add %arg200, %c_5 : tensor<i32>
    %10 = stablehlo.convert %9 : (tensor<i32>) -> tensor<f32>
    %cst_6 = stablehlo.constant dense<0.116355285> : tensor<f32>
    %11 = stablehlo.multiply %10, %cst_6 : tensor<f32>
    %12 = stablehlo.cosine %11 : tensor<f32>
    %13 = stablehlo.add %12, %cst : tensor<f32>
    %cst_7 = stablehlo.constant dense<1.500000e-05> : tensor<f32>
    %14 = stablehlo.multiply %13, %cst_7 : tensor<f32>
    %cst_8 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %15 = stablehlo.multiply %13, %cst_8 : tensor<f32>
    %16 = stablehlo.subtract %cst, %15 : tensor<f32>
    %cst_9 = stablehlo.constant dense<3.000000e-06> : tensor<f32>
    %17 = stablehlo.multiply %16, %cst_9 : tensor<f32>
    %18 = stablehlo.add %14, %17 : tensor<f32>
    %19 = stablehlo.select %1, %8, %18 : tensor<i1>, tensor<f32>
    %cst_10 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %20 = stablehlo.multiply %19, %cst_10 : tensor<f32>
    %21 = stablehlo.convert %20 : (tensor<f32>) -> tensor<bf16>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %23 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x128256xi32>
    %24 = stablehlo.iota dim = 2 : tensor<1x4096x128256xi32>
    %25 = stablehlo.compare  EQ, %23, %24 : (tensor<1x4096x128256xi32>, tensor<1x4096x128256xi32>) -> tensor<1x4096x128256xi1>
    %26 = stablehlo.convert %25 : (tensor<1x4096x128256xi1>) -> tensor<1x4096x128256xbf16>
    %27 = "stablehlo.all_gather"(%arg66) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<128256x144xbf16>) -> tensor<128256x576xbf16>
    %28 = stablehlo.dot_general %26, %27, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x128256xbf16>, tensor<128256x576xbf16>) -> tensor<1x4096x576xbf16>
    %29 = mhlo.copy %28 : tensor<1x4096x576xbf16>
    %30 = mhlo.copy %29 : tensor<1x4096x576xbf16>
    %31 = stablehlo.convert %30 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %32 = stablehlo.multiply %31, %31 : tensor<1x4096x576xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %33 = stablehlo.reduce(%32 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %cst_12 = stablehlo.constant dense<0.00173611112> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %35 = stablehlo.multiply %33, %34 : tensor<1x4096xf32>
    %cst_13 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %37 = stablehlo.add %35, %36 : tensor<1x4096xf32>
    %38 = stablehlo.rsqrt %37 : tensor<1x4096xf32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %40 = stablehlo.multiply %31, %39 : tensor<1x4096x576xf32>
    %41 = stablehlo.convert %40 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %42 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %43 = stablehlo.multiply %41, %42 : tensor<1x4096x576xbf16>
    %44 = mhlo.copy %43 : tensor<1x4096x576xbf16>
    %45 = mhlo.copy %44 : tensor<1x4096x576xbf16>
    %46 = "stablehlo.all_gather"(%arg10) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %47 = stablehlo.dot_general %45, %46, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %48 = mhlo.copy %47 : tensor<1x4096x9x128xbf16>
    %49 = stablehlo.reduce_precision %48, format = e8m7 : tensor<1x4096x9x128xbf16>
    %50 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %51 = stablehlo.broadcast_in_dim %arg203, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %52 = stablehlo.compare  EQ, %50, %51 : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %53 = stablehlo.iota dim = 1 : tensor<4096x4096xi32>
    %54 = stablehlo.iota dim = 0 : tensor<4096x4096xi32>
    %55 = stablehlo.compare  LE, %53, %54 : (tensor<4096x4096xi32>, tensor<4096x4096xi32>) -> tensor<4096x4096xi1>
    %56 = stablehlo.broadcast_in_dim %55, dims = [1, 2] : (tensor<4096x4096xi1>) -> tensor<1x4096x4096xi1>
    %57 = stablehlo.and %52, %56 : tensor<1x4096x4096xi1>
    %58 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x4096x4096xf32>
    %cst_14 = stablehlo.constant dense<-2.38197633E+38> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x4096x4096xf32>
    %60 = stablehlo.select %57, %58, %59 : tensor<1x4096x4096xi1>, tensor<1x4096x4096xf32>
    %cst_15 = stablehlo.constant dense<-1.19098816E+38> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x4096x4096xf32>
    %62 = stablehlo.compare  GE, %60, %61 : (tensor<1x4096x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xi1>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0, 3, 4] : (tensor<1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xi1>
    %64 = "stablehlo.all_gather"(%arg7) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 3, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %65 = stablehlo.dot_general %45, %64, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %66 = stablehlo.slice %65 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %67 = stablehlo.convert %arg202 : (tensor<1x4096xi32>) -> tensor<1x4096xf32>
    %68 = stablehlo.broadcast_in_dim %67, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x1x64xf32>
    %cst_16 = stablehlo.constant dense<5.000000e+05> : tensor<f32>
    %69 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %70 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_17 = stablehlo.constant dense<2> : tensor<i32>
    %71 = stablehlo.broadcast_in_dim %c_17, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %72 = stablehlo.multiply %70, %71 : tensor<64xi32>
    %73 = stablehlo.convert %72 : (tensor<64xi32>) -> tensor<64xf32>
    %cst_18 = stablehlo.constant dense<7.812500e-03> : tensor<f32>
    %74 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %75 = stablehlo.multiply %73, %74 : tensor<64xf32>
    %76 = stablehlo.power %69, %75 : tensor<64xf32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [3] : (tensor<64xf32>) -> tensor<1x4096x1x64xf32>
    %78 = stablehlo.divide %68, %77 : tensor<1x4096x1x64xf32>
    %79 = stablehlo.cosine %78 : tensor<1x4096x1x64xf32>
    %80 = stablehlo.convert %79 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %81 = stablehlo.reshape %80 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %82 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %83 = stablehlo.multiply %66, %82 : tensor<1x4096x9x64xbf16>
    %84 = stablehlo.slice %65 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %85 = stablehlo.sine %78 : tensor<1x4096x1x64xf32>
    %86 = stablehlo.convert %85 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %87 = stablehlo.reshape %86 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %88 = stablehlo.broadcast_in_dim %87, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %89 = stablehlo.multiply %84, %88 : tensor<1x4096x9x64xbf16>
    %90 = stablehlo.subtract %83, %89 : tensor<1x4096x9x64xbf16>
    %91 = stablehlo.multiply %84, %82 : tensor<1x4096x9x64xbf16>
    %92 = stablehlo.multiply %66, %88 : tensor<1x4096x9x64xbf16>
    %93 = stablehlo.add %91, %92 : tensor<1x4096x9x64xbf16>
    %94 = stablehlo.concatenate %90, %93, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %95 = mhlo.copy %94 : tensor<1x4096x9x128xbf16>
    %96 = stablehlo.reduce_precision %95, format = e8m7 : tensor<1x4096x9x128xbf16>
    %97 = "stablehlo.all_gather"(%arg9) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 4, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %98 = stablehlo.dot_general %45, %97, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %99 = stablehlo.slice %98 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %100 = stablehlo.multiply %99, %82 : tensor<1x4096x9x64xbf16>
    %101 = stablehlo.slice %98 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %102 = stablehlo.multiply %101, %88 : tensor<1x4096x9x64xbf16>
    %103 = stablehlo.subtract %100, %102 : tensor<1x4096x9x64xbf16>
    %104 = stablehlo.multiply %101, %82 : tensor<1x4096x9x64xbf16>
    %105 = stablehlo.multiply %99, %88 : tensor<1x4096x9x64xbf16>
    %106 = stablehlo.add %104, %105 : tensor<1x4096x9x64xbf16>
    %107 = stablehlo.concatenate %103, %106, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %108 = mhlo.copy %107 : tensor<1x4096x9x128xbf16>
    %109 = stablehlo.reduce_precision %108, format = e8m7 : tensor<1x4096x9x128xbf16>
    %110 = stablehlo.dot_general %96, %109, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %111 = stablehlo.reshape %110 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %112 = stablehlo.transpose %111, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %cst_19 = stablehlo.constant dense<-2.379320e+38> : tensor<bf16>
    %113 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<bf16>) -> tensor<1x9x1x4096x4096xbf16>
    %114 = stablehlo.select %63, %112, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %cst_20 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %115 = stablehlo.reduce(%114 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %116 = stablehlo.broadcast_in_dim %115, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %117 = stablehlo.subtract %114, %116 : tensor<1x9x1x4096x4096xbf16>
    %118 = stablehlo.exponential %117 : tensor<1x9x1x4096x4096xbf16>
    %119 = stablehlo.reshape %118 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %120 = stablehlo.dot_general %49, %119, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %121 = stablehlo.reshape %120 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %122 = stablehlo.transpose %121, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %123 = stablehlo.convert %119 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %124 = stablehlo.reduce(%123 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %125 = stablehlo.convert %124 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %126 = stablehlo.reshape %125 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %127 = stablehlo.transpose %126, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %128 = stablehlo.reshape %127 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %129 = stablehlo.broadcast_in_dim %128, dims = [0, 1, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %130 = stablehlo.divide %122, %129 {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : tensor<1x4096x9x1x128xbf16>
    %131 = stablehlo.reshape %130 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %132 = mhlo.copy %131 : tensor<1x4096x9x128xbf16>
    %133 = "stablehlo.all_gather"(%arg8) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 5, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %134 = stablehlo.dot_general %132, %133, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x576xbf16>
    %135 = stablehlo.reduce_precision %134, format = e8m7 : tensor<1x4096x576xbf16>
    %136 = mhlo.copy %135 : tensor<1x4096x576xbf16>
    %137 = stablehlo.add %30, %136 : tensor<1x4096x576xbf16>
    %138 = stablehlo.convert %137 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %139 = stablehlo.multiply %138, %138 : tensor<1x4096x576xf32>
    %140 = stablehlo.reduce(%139 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %141 = stablehlo.multiply %140, %34 : tensor<1x4096xf32>
    %142 = stablehlo.add %141, %36 : tensor<1x4096xf32>
    %143 = stablehlo.rsqrt %142 : tensor<1x4096xf32>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %145 = stablehlo.multiply %138, %144 : tensor<1x4096x576xf32>
    %146 = stablehlo.convert %145 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %147 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %148 = stablehlo.multiply %146, %147 : tensor<1x4096x576xbf16>
    %149 = mhlo.copy %148 : tensor<1x4096x576xbf16>
    %150 = "stablehlo.all_gather"(%arg2) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 6, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %151 = stablehlo.dot_general %149, %150, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %cst_21 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %152 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<bf16>) -> tensor<1x4096x2016xbf16>
    %153 = stablehlo.negate %151 : tensor<1x4096x2016xbf16>
    %154 = stablehlo.exponential %153 : tensor<1x4096x2016xbf16>
    %155 = stablehlo.add %154, %152 : tensor<1x4096x2016xbf16>
    %156 = stablehlo.divide %152, %155 : tensor<1x4096x2016xbf16>
    %157 = stablehlo.multiply %151, %156 : tensor<1x4096x2016xbf16>
    %158 = "stablehlo.all_gather"(%arg3) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 7, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %159 = stablehlo.dot_general %149, %158, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %160 = stablehlo.reduce_precision %159, format = e8m7 : tensor<1x4096x2016xbf16>
    %161 = stablehlo.multiply %157, %160 : tensor<1x4096x2016xbf16>
    %162 = mhlo.copy %161 : tensor<1x4096x2016xbf16>
    %163 = "stablehlo.all_gather"(%arg4) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 8, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %164 = stablehlo.dot_general %162, %163, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x576xbf16>
    %165 = mhlo.copy %164 : tensor<1x4096x576xbf16>
    %166 = stablehlo.add %165, %137 : tensor<1x4096x576xbf16>
    %167 = mhlo.copy %166 : tensor<1x4096x576xbf16>
    %168 = mhlo.copy %167 : tensor<1x4096x576xbf16>
    %169 = stablehlo.convert %168 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %170 = stablehlo.multiply %169, %169 : tensor<1x4096x576xf32>
    %171 = stablehlo.reduce(%170 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %172 = stablehlo.multiply %171, %34 : tensor<1x4096xf32>
    %173 = stablehlo.add %172, %36 : tensor<1x4096xf32>
    %174 = stablehlo.rsqrt %173 : tensor<1x4096xf32>
    %175 = stablehlo.broadcast_in_dim %174, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %176 = stablehlo.multiply %169, %175 : tensor<1x4096x576xf32>
    %177 = stablehlo.convert %176 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %178 = stablehlo.broadcast_in_dim %arg15, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %179 = stablehlo.multiply %177, %178 : tensor<1x4096x576xbf16>
    %180 = mhlo.copy %179 : tensor<1x4096x576xbf16>
    %181 = mhlo.copy %180 : tensor<1x4096x576xbf16>
    %182 = "stablehlo.all_gather"(%arg19) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 9, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %183 = stablehlo.dot_general %181, %182, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %184 = mhlo.copy %183 : tensor<1x4096x9x128xbf16>
    %185 = stablehlo.reduce_precision %184, format = e8m7 : tensor<1x4096x9x128xbf16>
    %186 = "stablehlo.all_gather"(%arg16) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 10, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %187 = stablehlo.dot_general %181, %186, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %188 = stablehlo.slice %187 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %189 = stablehlo.multiply %188, %82 : tensor<1x4096x9x64xbf16>
    %190 = stablehlo.slice %187 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %191 = stablehlo.multiply %190, %88 : tensor<1x4096x9x64xbf16>
    %192 = stablehlo.subtract %189, %191 : tensor<1x4096x9x64xbf16>
    %193 = stablehlo.multiply %190, %82 : tensor<1x4096x9x64xbf16>
    %194 = stablehlo.multiply %188, %88 : tensor<1x4096x9x64xbf16>
    %195 = stablehlo.add %193, %194 : tensor<1x4096x9x64xbf16>
    %196 = stablehlo.concatenate %192, %195, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %197 = mhlo.copy %196 : tensor<1x4096x9x128xbf16>
    %198 = stablehlo.reduce_precision %197, format = e8m7 : tensor<1x4096x9x128xbf16>
    %199 = "stablehlo.all_gather"(%arg18) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 11, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %200 = stablehlo.dot_general %181, %199, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %201 = stablehlo.slice %200 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %202 = stablehlo.multiply %201, %82 : tensor<1x4096x9x64xbf16>
    %203 = stablehlo.slice %200 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %204 = stablehlo.multiply %203, %88 : tensor<1x4096x9x64xbf16>
    %205 = stablehlo.subtract %202, %204 : tensor<1x4096x9x64xbf16>
    %206 = stablehlo.multiply %203, %82 : tensor<1x4096x9x64xbf16>
    %207 = stablehlo.multiply %201, %88 : tensor<1x4096x9x64xbf16>
    %208 = stablehlo.add %206, %207 : tensor<1x4096x9x64xbf16>
    %209 = stablehlo.concatenate %205, %208, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %210 = mhlo.copy %209 : tensor<1x4096x9x128xbf16>
    %211 = stablehlo.reduce_precision %210, format = e8m7 : tensor<1x4096x9x128xbf16>
    %212 = stablehlo.dot_general %198, %211, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %213 = stablehlo.reshape %212 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %214 = stablehlo.transpose %213, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %215 = stablehlo.select %63, %214, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %216 = stablehlo.reduce(%215 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %217 = stablehlo.broadcast_in_dim %216, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %218 = stablehlo.subtract %215, %217 : tensor<1x9x1x4096x4096xbf16>
    %219 = stablehlo.exponential %218 : tensor<1x9x1x4096x4096xbf16>
    %220 = stablehlo.reshape %219 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %221 = stablehlo.dot_general %185, %220, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %222 = stablehlo.reshape %221 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %223 = stablehlo.transpose %222, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %224 = stablehlo.convert %220 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %225 = stablehlo.reduce(%224 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %226 = stablehlo.convert %225 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %227 = stablehlo.reshape %226 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %228 = stablehlo.transpose %227, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %229 = stablehlo.reshape %228 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %231 = stablehlo.divide %223, %230 {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : tensor<1x4096x9x1x128xbf16>
    %232 = stablehlo.reshape %231 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %233 = mhlo.copy %232 : tensor<1x4096x9x128xbf16>
    %234 = "stablehlo.all_gather"(%arg17) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 12, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %235 = stablehlo.dot_general %233, %234, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x576xbf16>
    %236 = stablehlo.reduce_precision %235, format = e8m7 : tensor<1x4096x576xbf16>
    %237 = mhlo.copy %236 : tensor<1x4096x576xbf16>
    %238 = stablehlo.add %168, %237 : tensor<1x4096x576xbf16>
    %239 = stablehlo.convert %238 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %240 = stablehlo.multiply %239, %239 : tensor<1x4096x576xf32>
    %241 = stablehlo.reduce(%240 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %242 = stablehlo.multiply %241, %34 : tensor<1x4096xf32>
    %243 = stablehlo.add %242, %36 : tensor<1x4096xf32>
    %244 = stablehlo.rsqrt %243 : tensor<1x4096xf32>
    %245 = stablehlo.broadcast_in_dim %244, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %246 = stablehlo.multiply %239, %245 : tensor<1x4096x576xf32>
    %247 = stablehlo.convert %246 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %248 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %249 = stablehlo.multiply %247, %248 : tensor<1x4096x576xbf16>
    %250 = mhlo.copy %249 : tensor<1x4096x576xbf16>
    %251 = "stablehlo.all_gather"(%arg11) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 13, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %252 = stablehlo.dot_general %250, %251, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %253 = stablehlo.negate %252 : tensor<1x4096x2016xbf16>
    %254 = stablehlo.exponential %253 : tensor<1x4096x2016xbf16>
    %255 = stablehlo.add %254, %152 : tensor<1x4096x2016xbf16>
    %256 = stablehlo.divide %152, %255 : tensor<1x4096x2016xbf16>
    %257 = stablehlo.multiply %252, %256 : tensor<1x4096x2016xbf16>
    %258 = "stablehlo.all_gather"(%arg12) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 14, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %259 = stablehlo.dot_general %250, %258, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %260 = stablehlo.reduce_precision %259, format = e8m7 : tensor<1x4096x2016xbf16>
    %261 = stablehlo.multiply %257, %260 : tensor<1x4096x2016xbf16>
    %262 = mhlo.copy %261 : tensor<1x4096x2016xbf16>
    %263 = "stablehlo.all_gather"(%arg13) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 15, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %264 = stablehlo.dot_general %262, %263, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x576xbf16>
    %265 = mhlo.copy %264 : tensor<1x4096x576xbf16>
    %266 = stablehlo.add %265, %238 : tensor<1x4096x576xbf16>
    %267 = mhlo.copy %266 : tensor<1x4096x576xbf16>
    %268 = mhlo.copy %267 : tensor<1x4096x576xbf16>
    %269 = stablehlo.convert %268 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %270 = stablehlo.multiply %269, %269 : tensor<1x4096x576xf32>
    %271 = stablehlo.reduce(%270 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %272 = stablehlo.multiply %271, %34 : tensor<1x4096xf32>
    %273 = stablehlo.add %272, %36 : tensor<1x4096xf32>
    %274 = stablehlo.rsqrt %273 : tensor<1x4096xf32>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %276 = stablehlo.multiply %269, %275 : tensor<1x4096x576xf32>
    %277 = stablehlo.convert %276 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %278 = stablehlo.broadcast_in_dim %arg24, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %279 = stablehlo.multiply %277, %278 : tensor<1x4096x576xbf16>
    %280 = mhlo.copy %279 : tensor<1x4096x576xbf16>
    %281 = mhlo.copy %280 : tensor<1x4096x576xbf16>
    %282 = "stablehlo.all_gather"(%arg28) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 16, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %283 = stablehlo.dot_general %281, %282, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %284 = mhlo.copy %283 : tensor<1x4096x9x128xbf16>
    %285 = stablehlo.reduce_precision %284, format = e8m7 : tensor<1x4096x9x128xbf16>
    %286 = "stablehlo.all_gather"(%arg25) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 17, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %287 = stablehlo.dot_general %281, %286, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %288 = stablehlo.slice %287 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %289 = stablehlo.multiply %288, %82 : tensor<1x4096x9x64xbf16>
    %290 = stablehlo.slice %287 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %291 = stablehlo.multiply %290, %88 : tensor<1x4096x9x64xbf16>
    %292 = stablehlo.subtract %289, %291 : tensor<1x4096x9x64xbf16>
    %293 = stablehlo.multiply %290, %82 : tensor<1x4096x9x64xbf16>
    %294 = stablehlo.multiply %288, %88 : tensor<1x4096x9x64xbf16>
    %295 = stablehlo.add %293, %294 : tensor<1x4096x9x64xbf16>
    %296 = stablehlo.concatenate %292, %295, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %297 = mhlo.copy %296 : tensor<1x4096x9x128xbf16>
    %298 = stablehlo.reduce_precision %297, format = e8m7 : tensor<1x4096x9x128xbf16>
    %299 = "stablehlo.all_gather"(%arg27) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 18, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %300 = stablehlo.dot_general %281, %299, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %301 = stablehlo.slice %300 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %302 = stablehlo.multiply %301, %82 : tensor<1x4096x9x64xbf16>
    %303 = stablehlo.slice %300 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %304 = stablehlo.multiply %303, %88 : tensor<1x4096x9x64xbf16>
    %305 = stablehlo.subtract %302, %304 : tensor<1x4096x9x64xbf16>
    %306 = stablehlo.multiply %303, %82 : tensor<1x4096x9x64xbf16>
    %307 = stablehlo.multiply %301, %88 : tensor<1x4096x9x64xbf16>
    %308 = stablehlo.add %306, %307 : tensor<1x4096x9x64xbf16>
    %309 = stablehlo.concatenate %305, %308, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %310 = mhlo.copy %309 : tensor<1x4096x9x128xbf16>
    %311 = stablehlo.reduce_precision %310, format = e8m7 : tensor<1x4096x9x128xbf16>
    %312 = stablehlo.dot_general %298, %311, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %313 = stablehlo.reshape %312 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %314 = stablehlo.transpose %313, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %315 = stablehlo.select %63, %314, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %316 = stablehlo.reduce(%315 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %318 = stablehlo.subtract %315, %317 : tensor<1x9x1x4096x4096xbf16>
    %319 = stablehlo.exponential %318 : tensor<1x9x1x4096x4096xbf16>
    %320 = stablehlo.reshape %319 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %321 = stablehlo.dot_general %285, %320, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %322 = stablehlo.reshape %321 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %323 = stablehlo.transpose %322, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %324 = stablehlo.convert %320 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %325 = stablehlo.reduce(%324 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %326 = stablehlo.convert %325 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %327 = stablehlo.reshape %326 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %328 = stablehlo.transpose %327, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %329 = stablehlo.reshape %328 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %330 = stablehlo.broadcast_in_dim %329, dims = [0, 1, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %331 = stablehlo.divide %323, %330 {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : tensor<1x4096x9x1x128xbf16>
    %332 = stablehlo.reshape %331 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %333 = mhlo.copy %332 : tensor<1x4096x9x128xbf16>
    %334 = "stablehlo.all_gather"(%arg26) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 19, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %335 = stablehlo.dot_general %333, %334, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x576xbf16>
    %336 = stablehlo.reduce_precision %335, format = e8m7 : tensor<1x4096x576xbf16>
    %337 = mhlo.copy %336 : tensor<1x4096x576xbf16>
    %338 = stablehlo.add %268, %337 : tensor<1x4096x576xbf16>
    %339 = stablehlo.convert %338 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %340 = stablehlo.multiply %339, %339 : tensor<1x4096x576xf32>
    %341 = stablehlo.reduce(%340 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %342 = stablehlo.multiply %341, %34 : tensor<1x4096xf32>
    %343 = stablehlo.add %342, %36 : tensor<1x4096xf32>
    %344 = stablehlo.rsqrt %343 : tensor<1x4096xf32>
    %345 = stablehlo.broadcast_in_dim %344, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %346 = stablehlo.multiply %339, %345 : tensor<1x4096x576xf32>
    %347 = stablehlo.convert %346 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %348 = stablehlo.broadcast_in_dim %arg23, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %349 = stablehlo.multiply %347, %348 : tensor<1x4096x576xbf16>
    %350 = mhlo.copy %349 : tensor<1x4096x576xbf16>
    %351 = "stablehlo.all_gather"(%arg20) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 20, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %352 = stablehlo.dot_general %350, %351, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %353 = stablehlo.negate %352 : tensor<1x4096x2016xbf16>
    %354 = stablehlo.exponential %353 : tensor<1x4096x2016xbf16>
    %355 = stablehlo.add %354, %152 : tensor<1x4096x2016xbf16>
    %356 = stablehlo.divide %152, %355 : tensor<1x4096x2016xbf16>
    %357 = stablehlo.multiply %352, %356 : tensor<1x4096x2016xbf16>
    %358 = "stablehlo.all_gather"(%arg21) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 21, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %359 = stablehlo.dot_general %350, %358, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %360 = stablehlo.reduce_precision %359, format = e8m7 : tensor<1x4096x2016xbf16>
    %361 = stablehlo.multiply %357, %360 : tensor<1x4096x2016xbf16>
    %362 = mhlo.copy %361 : tensor<1x4096x2016xbf16>
    %363 = "stablehlo.all_gather"(%arg22) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 22, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %364 = stablehlo.dot_general %362, %363, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x576xbf16>
    %365 = mhlo.copy %364 : tensor<1x4096x576xbf16>
    %366 = stablehlo.add %365, %338 : tensor<1x4096x576xbf16>
    %367 = mhlo.copy %366 : tensor<1x4096x576xbf16>
    %368 = mhlo.copy %367 : tensor<1x4096x576xbf16>
    %369 = stablehlo.convert %368 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %370 = stablehlo.multiply %369, %369 : tensor<1x4096x576xf32>
    %371 = stablehlo.reduce(%370 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %372 = stablehlo.multiply %371, %34 : tensor<1x4096xf32>
    %373 = stablehlo.add %372, %36 : tensor<1x4096xf32>
    %374 = stablehlo.rsqrt %373 : tensor<1x4096xf32>
    %375 = stablehlo.broadcast_in_dim %374, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %376 = stablehlo.multiply %369, %375 : tensor<1x4096x576xf32>
    %377 = stablehlo.convert %376 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %378 = stablehlo.broadcast_in_dim %arg33, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %379 = stablehlo.multiply %377, %378 : tensor<1x4096x576xbf16>
    %380 = mhlo.copy %379 : tensor<1x4096x576xbf16>
    %381 = mhlo.copy %380 : tensor<1x4096x576xbf16>
    %382 = "stablehlo.all_gather"(%arg37) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 23, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %383 = stablehlo.dot_general %381, %382, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %384 = mhlo.copy %383 : tensor<1x4096x9x128xbf16>
    %385 = stablehlo.reduce_precision %384, format = e8m7 : tensor<1x4096x9x128xbf16>
    %386 = "stablehlo.all_gather"(%arg34) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 24, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %387 = stablehlo.dot_general %381, %386, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %388 = stablehlo.slice %387 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %389 = stablehlo.multiply %388, %82 : tensor<1x4096x9x64xbf16>
    %390 = stablehlo.slice %387 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %391 = stablehlo.multiply %390, %88 : tensor<1x4096x9x64xbf16>
    %392 = stablehlo.subtract %389, %391 : tensor<1x4096x9x64xbf16>
    %393 = stablehlo.multiply %390, %82 : tensor<1x4096x9x64xbf16>
    %394 = stablehlo.multiply %388, %88 : tensor<1x4096x9x64xbf16>
    %395 = stablehlo.add %393, %394 : tensor<1x4096x9x64xbf16>
    %396 = stablehlo.concatenate %392, %395, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %397 = mhlo.copy %396 : tensor<1x4096x9x128xbf16>
    %398 = stablehlo.reduce_precision %397, format = e8m7 : tensor<1x4096x9x128xbf16>
    %399 = "stablehlo.all_gather"(%arg36) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 25, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %400 = stablehlo.dot_general %381, %399, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %401 = stablehlo.slice %400 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %402 = stablehlo.multiply %401, %82 : tensor<1x4096x9x64xbf16>
    %403 = stablehlo.slice %400 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %404 = stablehlo.multiply %403, %88 : tensor<1x4096x9x64xbf16>
    %405 = stablehlo.subtract %402, %404 : tensor<1x4096x9x64xbf16>
    %406 = stablehlo.multiply %403, %82 : tensor<1x4096x9x64xbf16>
    %407 = stablehlo.multiply %401, %88 : tensor<1x4096x9x64xbf16>
    %408 = stablehlo.add %406, %407 : tensor<1x4096x9x64xbf16>
    %409 = stablehlo.concatenate %405, %408, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %410 = mhlo.copy %409 : tensor<1x4096x9x128xbf16>
    %411 = stablehlo.reduce_precision %410, format = e8m7 : tensor<1x4096x9x128xbf16>
    %412 = stablehlo.dot_general %398, %411, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %413 = stablehlo.reshape %412 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %414 = stablehlo.transpose %413, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %415 = stablehlo.select %63, %414, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %416 = stablehlo.reduce(%415 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %417 = stablehlo.broadcast_in_dim %416, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %418 = stablehlo.subtract %415, %417 : tensor<1x9x1x4096x4096xbf16>
    %419 = stablehlo.exponential %418 : tensor<1x9x1x4096x4096xbf16>
    %420 = stablehlo.reshape %419 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %421 = stablehlo.dot_general %385, %420, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %422 = stablehlo.reshape %421 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %423 = stablehlo.transpose %422, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %424 = stablehlo.convert %420 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %425 = stablehlo.reduce(%424 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %426 = stablehlo.convert %425 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %427 = stablehlo.reshape %426 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %428 = stablehlo.transpose %427, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %429 = stablehlo.reshape %428 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %430 = stablehlo.broadcast_in_dim %429, dims = [0, 1, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %431 = stablehlo.divide %423, %430 {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : tensor<1x4096x9x1x128xbf16>
    %432 = stablehlo.reshape %431 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %433 = mhlo.copy %432 : tensor<1x4096x9x128xbf16>
    %434 = "stablehlo.all_gather"(%arg35) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 26, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %435 = stablehlo.dot_general %433, %434, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x576xbf16>
    %436 = stablehlo.reduce_precision %435, format = e8m7 : tensor<1x4096x576xbf16>
    %437 = mhlo.copy %436 : tensor<1x4096x576xbf16>
    %438 = stablehlo.add %368, %437 : tensor<1x4096x576xbf16>
    %439 = stablehlo.convert %438 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %440 = stablehlo.multiply %439, %439 : tensor<1x4096x576xf32>
    %441 = stablehlo.reduce(%440 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %442 = stablehlo.multiply %441, %34 : tensor<1x4096xf32>
    %443 = stablehlo.add %442, %36 : tensor<1x4096xf32>
    %444 = stablehlo.rsqrt %443 : tensor<1x4096xf32>
    %445 = stablehlo.broadcast_in_dim %444, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %446 = stablehlo.multiply %439, %445 : tensor<1x4096x576xf32>
    %447 = stablehlo.convert %446 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %448 = stablehlo.broadcast_in_dim %arg32, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %449 = stablehlo.multiply %447, %448 : tensor<1x4096x576xbf16>
    %450 = mhlo.copy %449 : tensor<1x4096x576xbf16>
    %451 = "stablehlo.all_gather"(%arg29) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 27, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %452 = stablehlo.dot_general %450, %451, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %453 = stablehlo.negate %452 : tensor<1x4096x2016xbf16>
    %454 = stablehlo.exponential %453 : tensor<1x4096x2016xbf16>
    %455 = stablehlo.add %454, %152 : tensor<1x4096x2016xbf16>
    %456 = stablehlo.divide %152, %455 : tensor<1x4096x2016xbf16>
    %457 = stablehlo.multiply %452, %456 : tensor<1x4096x2016xbf16>
    %458 = "stablehlo.all_gather"(%arg30) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 28, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %459 = stablehlo.dot_general %450, %458, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %460 = stablehlo.reduce_precision %459, format = e8m7 : tensor<1x4096x2016xbf16>
    %461 = stablehlo.multiply %457, %460 : tensor<1x4096x2016xbf16>
    %462 = mhlo.copy %461 : tensor<1x4096x2016xbf16>
    %463 = "stablehlo.all_gather"(%arg31) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 29, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %464 = stablehlo.dot_general %462, %463, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x576xbf16>
    %465 = mhlo.copy %464 : tensor<1x4096x576xbf16>
    %466 = stablehlo.add %465, %438 : tensor<1x4096x576xbf16>
    %467 = mhlo.copy %466 : tensor<1x4096x576xbf16>
    %468 = mhlo.copy %467 : tensor<1x4096x576xbf16>
    %469 = stablehlo.convert %468 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %470 = stablehlo.multiply %469, %469 : tensor<1x4096x576xf32>
    %471 = stablehlo.reduce(%470 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %472 = stablehlo.multiply %471, %34 : tensor<1x4096xf32>
    %473 = stablehlo.add %472, %36 : tensor<1x4096xf32>
    %474 = stablehlo.rsqrt %473 : tensor<1x4096xf32>
    %475 = stablehlo.broadcast_in_dim %474, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %476 = stablehlo.multiply %469, %475 : tensor<1x4096x576xf32>
    %477 = stablehlo.convert %476 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %478 = stablehlo.broadcast_in_dim %arg42, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %479 = stablehlo.multiply %477, %478 : tensor<1x4096x576xbf16>
    %480 = mhlo.copy %479 : tensor<1x4096x576xbf16>
    %481 = mhlo.copy %480 : tensor<1x4096x576xbf16>
    %482 = "stablehlo.all_gather"(%arg46) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 30, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %483 = stablehlo.dot_general %481, %482, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %484 = mhlo.copy %483 : tensor<1x4096x9x128xbf16>
    %485 = stablehlo.reduce_precision %484, format = e8m7 : tensor<1x4096x9x128xbf16>
    %486 = "stablehlo.all_gather"(%arg43) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 31, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %487 = stablehlo.dot_general %481, %486, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %488 = stablehlo.slice %487 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %489 = stablehlo.multiply %488, %82 : tensor<1x4096x9x64xbf16>
    %490 = stablehlo.slice %487 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %491 = stablehlo.multiply %490, %88 : tensor<1x4096x9x64xbf16>
    %492 = stablehlo.subtract %489, %491 : tensor<1x4096x9x64xbf16>
    %493 = stablehlo.multiply %490, %82 : tensor<1x4096x9x64xbf16>
    %494 = stablehlo.multiply %488, %88 : tensor<1x4096x9x64xbf16>
    %495 = stablehlo.add %493, %494 : tensor<1x4096x9x64xbf16>
    %496 = stablehlo.concatenate %492, %495, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %497 = mhlo.copy %496 : tensor<1x4096x9x128xbf16>
    %498 = stablehlo.reduce_precision %497, format = e8m7 : tensor<1x4096x9x128xbf16>
    %499 = "stablehlo.all_gather"(%arg45) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 32, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %500 = stablehlo.dot_general %481, %499, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %501 = stablehlo.slice %500 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %502 = stablehlo.multiply %501, %82 : tensor<1x4096x9x64xbf16>
    %503 = stablehlo.slice %500 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %504 = stablehlo.multiply %503, %88 : tensor<1x4096x9x64xbf16>
    %505 = stablehlo.subtract %502, %504 : tensor<1x4096x9x64xbf16>
    %506 = stablehlo.multiply %503, %82 : tensor<1x4096x9x64xbf16>
    %507 = stablehlo.multiply %501, %88 : tensor<1x4096x9x64xbf16>
    %508 = stablehlo.add %506, %507 : tensor<1x4096x9x64xbf16>
    %509 = stablehlo.concatenate %505, %508, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %510 = mhlo.copy %509 : tensor<1x4096x9x128xbf16>
    %511 = stablehlo.reduce_precision %510, format = e8m7 : tensor<1x4096x9x128xbf16>
    %512 = stablehlo.dot_general %498, %511, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %513 = stablehlo.reshape %512 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %514 = stablehlo.transpose %513, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %515 = stablehlo.select %63, %514, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %516 = stablehlo.reduce(%515 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %517 = stablehlo.broadcast_in_dim %516, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %518 = stablehlo.subtract %515, %517 : tensor<1x9x1x4096x4096xbf16>
    %519 = stablehlo.exponential %518 : tensor<1x9x1x4096x4096xbf16>
    %520 = stablehlo.reshape %519 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %521 = stablehlo.dot_general %485, %520, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %522 = stablehlo.reshape %521 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %523 = stablehlo.transpose %522, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %524 = stablehlo.convert %520 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %525 = stablehlo.reduce(%524 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %526 = stablehlo.convert %525 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %527 = stablehlo.reshape %526 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %528 = stablehlo.transpose %527, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %529 = stablehlo.reshape %528 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %530 = stablehlo.broadcast_in_dim %529, dims = [0, 1, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %531 = stablehlo.divide %523, %530 {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : tensor<1x4096x9x1x128xbf16>
    %532 = stablehlo.reshape %531 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %533 = mhlo.copy %532 : tensor<1x4096x9x128xbf16>
    %534 = "stablehlo.all_gather"(%arg44) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 33, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %535 = stablehlo.dot_general %533, %534, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x576xbf16>
    %536 = stablehlo.reduce_precision %535, format = e8m7 : tensor<1x4096x576xbf16>
    %537 = mhlo.copy %536 : tensor<1x4096x576xbf16>
    %538 = stablehlo.add %468, %537 : tensor<1x4096x576xbf16>
    %539 = stablehlo.convert %538 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %540 = stablehlo.multiply %539, %539 : tensor<1x4096x576xf32>
    %541 = stablehlo.reduce(%540 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %542 = stablehlo.multiply %541, %34 : tensor<1x4096xf32>
    %543 = stablehlo.add %542, %36 : tensor<1x4096xf32>
    %544 = stablehlo.rsqrt %543 : tensor<1x4096xf32>
    %545 = stablehlo.broadcast_in_dim %544, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %546 = stablehlo.multiply %539, %545 : tensor<1x4096x576xf32>
    %547 = stablehlo.convert %546 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %548 = stablehlo.broadcast_in_dim %arg41, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %549 = stablehlo.multiply %547, %548 : tensor<1x4096x576xbf16>
    %550 = mhlo.copy %549 : tensor<1x4096x576xbf16>
    %551 = "stablehlo.all_gather"(%arg38) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 34, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %552 = stablehlo.dot_general %550, %551, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %553 = stablehlo.negate %552 : tensor<1x4096x2016xbf16>
    %554 = stablehlo.exponential %553 : tensor<1x4096x2016xbf16>
    %555 = stablehlo.add %554, %152 : tensor<1x4096x2016xbf16>
    %556 = stablehlo.divide %152, %555 : tensor<1x4096x2016xbf16>
    %557 = stablehlo.multiply %552, %556 : tensor<1x4096x2016xbf16>
    %558 = "stablehlo.all_gather"(%arg39) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 35, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %559 = stablehlo.dot_general %550, %558, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %560 = stablehlo.reduce_precision %559, format = e8m7 : tensor<1x4096x2016xbf16>
    %561 = stablehlo.multiply %557, %560 : tensor<1x4096x2016xbf16>
    %562 = mhlo.copy %561 : tensor<1x4096x2016xbf16>
    %563 = "stablehlo.all_gather"(%arg40) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 36, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %564 = stablehlo.dot_general %562, %563, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x576xbf16>
    %565 = mhlo.copy %564 : tensor<1x4096x576xbf16>
    %566 = stablehlo.add %565, %538 : tensor<1x4096x576xbf16>
    %567 = mhlo.copy %566 : tensor<1x4096x576xbf16>
    %568 = mhlo.copy %567 : tensor<1x4096x576xbf16>
    %569 = stablehlo.convert %568 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %570 = stablehlo.multiply %569, %569 : tensor<1x4096x576xf32>
    %571 = stablehlo.reduce(%570 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %572 = stablehlo.multiply %571, %34 : tensor<1x4096xf32>
    %573 = stablehlo.add %572, %36 : tensor<1x4096xf32>
    %574 = stablehlo.rsqrt %573 : tensor<1x4096xf32>
    %575 = stablehlo.broadcast_in_dim %574, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %576 = stablehlo.multiply %569, %575 : tensor<1x4096x576xf32>
    %577 = stablehlo.convert %576 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %578 = stablehlo.broadcast_in_dim %arg51, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %579 = stablehlo.multiply %577, %578 : tensor<1x4096x576xbf16>
    %580 = mhlo.copy %579 : tensor<1x4096x576xbf16>
    %581 = mhlo.copy %580 : tensor<1x4096x576xbf16>
    %582 = "stablehlo.all_gather"(%arg55) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 37, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %583 = stablehlo.dot_general %581, %582, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %584 = mhlo.copy %583 : tensor<1x4096x9x128xbf16>
    %585 = stablehlo.reduce_precision %584, format = e8m7 : tensor<1x4096x9x128xbf16>
    %586 = "stablehlo.all_gather"(%arg52) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 38, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %587 = stablehlo.dot_general %581, %586, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %588 = stablehlo.slice %587 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %589 = stablehlo.multiply %588, %82 : tensor<1x4096x9x64xbf16>
    %590 = stablehlo.slice %587 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %591 = stablehlo.multiply %590, %88 : tensor<1x4096x9x64xbf16>
    %592 = stablehlo.subtract %589, %591 : tensor<1x4096x9x64xbf16>
    %593 = stablehlo.multiply %590, %82 : tensor<1x4096x9x64xbf16>
    %594 = stablehlo.multiply %588, %88 : tensor<1x4096x9x64xbf16>
    %595 = stablehlo.add %593, %594 : tensor<1x4096x9x64xbf16>
    %596 = stablehlo.concatenate %592, %595, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %597 = mhlo.copy %596 : tensor<1x4096x9x128xbf16>
    %598 = stablehlo.reduce_precision %597, format = e8m7 : tensor<1x4096x9x128xbf16>
    %599 = "stablehlo.all_gather"(%arg54) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 39, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %600 = stablehlo.dot_general %581, %599, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %601 = stablehlo.slice %600 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %602 = stablehlo.multiply %601, %82 : tensor<1x4096x9x64xbf16>
    %603 = stablehlo.slice %600 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %604 = stablehlo.multiply %603, %88 : tensor<1x4096x9x64xbf16>
    %605 = stablehlo.subtract %602, %604 : tensor<1x4096x9x64xbf16>
    %606 = stablehlo.multiply %603, %82 : tensor<1x4096x9x64xbf16>
    %607 = stablehlo.multiply %601, %88 : tensor<1x4096x9x64xbf16>
    %608 = stablehlo.add %606, %607 : tensor<1x4096x9x64xbf16>
    %609 = stablehlo.concatenate %605, %608, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %610 = mhlo.copy %609 : tensor<1x4096x9x128xbf16>
    %611 = stablehlo.reduce_precision %610, format = e8m7 : tensor<1x4096x9x128xbf16>
    %612 = stablehlo.dot_general %598, %611, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %613 = stablehlo.reshape %612 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %614 = stablehlo.transpose %613, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %615 = stablehlo.select %63, %614, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %616 = stablehlo.reduce(%615 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %617 = stablehlo.broadcast_in_dim %616, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %618 = stablehlo.subtract %615, %617 : tensor<1x9x1x4096x4096xbf16>
    %619 = stablehlo.exponential %618 : tensor<1x9x1x4096x4096xbf16>
    %620 = stablehlo.reshape %619 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %621 = stablehlo.dot_general %585, %620, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %622 = stablehlo.reshape %621 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %623 = stablehlo.transpose %622, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %624 = stablehlo.convert %620 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %625 = stablehlo.reduce(%624 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %626 = stablehlo.convert %625 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %627 = stablehlo.reshape %626 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %628 = stablehlo.transpose %627, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %629 = stablehlo.reshape %628 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %630 = stablehlo.broadcast_in_dim %629, dims = [0, 1, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %631 = stablehlo.divide %623, %630 {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : tensor<1x4096x9x1x128xbf16>
    %632 = stablehlo.reshape %631 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %633 = mhlo.copy %632 : tensor<1x4096x9x128xbf16>
    %634 = "stablehlo.all_gather"(%arg53) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 40, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %635 = stablehlo.dot_general %633, %634, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x576xbf16>
    %636 = stablehlo.reduce_precision %635, format = e8m7 : tensor<1x4096x576xbf16>
    %637 = mhlo.copy %636 : tensor<1x4096x576xbf16>
    %638 = stablehlo.add %568, %637 : tensor<1x4096x576xbf16>
    %639 = stablehlo.convert %638 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %640 = stablehlo.multiply %639, %639 : tensor<1x4096x576xf32>
    %641 = stablehlo.reduce(%640 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %642 = stablehlo.multiply %641, %34 : tensor<1x4096xf32>
    %643 = stablehlo.add %642, %36 : tensor<1x4096xf32>
    %644 = stablehlo.rsqrt %643 : tensor<1x4096xf32>
    %645 = stablehlo.broadcast_in_dim %644, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %646 = stablehlo.multiply %639, %645 : tensor<1x4096x576xf32>
    %647 = stablehlo.convert %646 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %648 = stablehlo.broadcast_in_dim %arg50, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %649 = stablehlo.multiply %647, %648 : tensor<1x4096x576xbf16>
    %650 = mhlo.copy %649 : tensor<1x4096x576xbf16>
    %651 = "stablehlo.all_gather"(%arg47) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 41, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %652 = stablehlo.dot_general %650, %651, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %653 = stablehlo.negate %652 : tensor<1x4096x2016xbf16>
    %654 = stablehlo.exponential %653 : tensor<1x4096x2016xbf16>
    %655 = stablehlo.add %654, %152 : tensor<1x4096x2016xbf16>
    %656 = stablehlo.divide %152, %655 : tensor<1x4096x2016xbf16>
    %657 = stablehlo.multiply %652, %656 : tensor<1x4096x2016xbf16>
    %658 = "stablehlo.all_gather"(%arg48) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 42, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %659 = stablehlo.dot_general %650, %658, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %660 = stablehlo.reduce_precision %659, format = e8m7 : tensor<1x4096x2016xbf16>
    %661 = stablehlo.multiply %657, %660 : tensor<1x4096x2016xbf16>
    %662 = mhlo.copy %661 : tensor<1x4096x2016xbf16>
    %663 = "stablehlo.all_gather"(%arg49) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 43, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %664 = stablehlo.dot_general %662, %663, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x576xbf16>
    %665 = mhlo.copy %664 : tensor<1x4096x576xbf16>
    %666 = stablehlo.add %665, %638 : tensor<1x4096x576xbf16>
    %667 = mhlo.copy %666 : tensor<1x4096x576xbf16>
    %668 = mhlo.copy %667 : tensor<1x4096x576xbf16>
    %669 = stablehlo.convert %668 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %670 = stablehlo.multiply %669, %669 : tensor<1x4096x576xf32>
    %671 = stablehlo.reduce(%670 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %672 = stablehlo.multiply %671, %34 : tensor<1x4096xf32>
    %673 = stablehlo.add %672, %36 : tensor<1x4096xf32>
    %674 = stablehlo.rsqrt %673 : tensor<1x4096xf32>
    %675 = stablehlo.broadcast_in_dim %674, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %676 = stablehlo.multiply %669, %675 : tensor<1x4096x576xf32>
    %677 = stablehlo.convert %676 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %678 = stablehlo.broadcast_in_dim %arg60, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %679 = stablehlo.multiply %677, %678 : tensor<1x4096x576xbf16>
    %680 = mhlo.copy %679 : tensor<1x4096x576xbf16>
    %681 = mhlo.copy %680 : tensor<1x4096x576xbf16>
    %682 = "stablehlo.all_gather"(%arg64) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 44, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %683 = stablehlo.dot_general %681, %682, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %684 = mhlo.copy %683 : tensor<1x4096x9x128xbf16>
    %685 = stablehlo.reduce_precision %684, format = e8m7 : tensor<1x4096x9x128xbf16>
    %686 = "stablehlo.all_gather"(%arg61) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 45, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %687 = stablehlo.dot_general %681, %686, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %688 = stablehlo.slice %687 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %689 = stablehlo.multiply %688, %82 : tensor<1x4096x9x64xbf16>
    %690 = stablehlo.slice %687 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %691 = stablehlo.multiply %690, %88 : tensor<1x4096x9x64xbf16>
    %692 = stablehlo.subtract %689, %691 : tensor<1x4096x9x64xbf16>
    %693 = stablehlo.multiply %690, %82 : tensor<1x4096x9x64xbf16>
    %694 = stablehlo.multiply %688, %88 : tensor<1x4096x9x64xbf16>
    %695 = stablehlo.add %693, %694 : tensor<1x4096x9x64xbf16>
    %696 = stablehlo.concatenate %692, %695, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %697 = mhlo.copy %696 : tensor<1x4096x9x128xbf16>
    %698 = stablehlo.reduce_precision %697, format = e8m7 : tensor<1x4096x9x128xbf16>
    %699 = "stablehlo.all_gather"(%arg63) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 46, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %700 = stablehlo.dot_general %681, %699, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %701 = stablehlo.slice %700 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %702 = stablehlo.multiply %701, %82 : tensor<1x4096x9x64xbf16>
    %703 = stablehlo.slice %700 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %704 = stablehlo.multiply %703, %88 : tensor<1x4096x9x64xbf16>
    %705 = stablehlo.subtract %702, %704 : tensor<1x4096x9x64xbf16>
    %706 = stablehlo.multiply %703, %82 : tensor<1x4096x9x64xbf16>
    %707 = stablehlo.multiply %701, %88 : tensor<1x4096x9x64xbf16>
    %708 = stablehlo.add %706, %707 : tensor<1x4096x9x64xbf16>
    %709 = stablehlo.concatenate %705, %708, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %710 = mhlo.copy %709 : tensor<1x4096x9x128xbf16>
    %711 = stablehlo.reduce_precision %710, format = e8m7 : tensor<1x4096x9x128xbf16>
    %712 = stablehlo.dot_general %698, %711, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %713 = stablehlo.reshape %712 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %714 = stablehlo.transpose %713, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %715 = stablehlo.select %63, %714, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %716 = stablehlo.reduce(%715 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %717 = stablehlo.broadcast_in_dim %716, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %718 = stablehlo.subtract %715, %717 : tensor<1x9x1x4096x4096xbf16>
    %719 = stablehlo.exponential %718 : tensor<1x9x1x4096x4096xbf16>
    %720 = stablehlo.reshape %719 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %721 = stablehlo.dot_general %685, %720, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %722 = stablehlo.reshape %721 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %723 = stablehlo.transpose %722, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %724 = stablehlo.convert %720 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %725 = stablehlo.reduce(%724 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %726 = stablehlo.convert %725 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %727 = stablehlo.reshape %726 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %728 = stablehlo.transpose %727, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %729 = stablehlo.reshape %728 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %730 = stablehlo.broadcast_in_dim %729, dims = [0, 1, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %731 = stablehlo.divide %723, %730 {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : tensor<1x4096x9x1x128xbf16>
    %732 = stablehlo.reshape %731 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %733 = mhlo.copy %732 : tensor<1x4096x9x128xbf16>
    %734 = "stablehlo.all_gather"(%arg62) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 47, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %735 = stablehlo.dot_general %733, %734, contracting_dims = [2, 3] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x576xbf16>
    %736 = stablehlo.reduce_precision %735, format = e8m7 : tensor<1x4096x576xbf16>
    %737 = mhlo.copy %736 : tensor<1x4096x576xbf16>
    %738 = stablehlo.add %668, %737 : tensor<1x4096x576xbf16>
    %739 = stablehlo.convert %738 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %740 = stablehlo.multiply %739, %739 : tensor<1x4096x576xf32>
    %741 = stablehlo.reduce(%740 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %742 = stablehlo.multiply %741, %34 : tensor<1x4096xf32>
    %743 = stablehlo.add %742, %36 : tensor<1x4096xf32>
    %744 = stablehlo.rsqrt %743 : tensor<1x4096xf32>
    %745 = stablehlo.broadcast_in_dim %744, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %746 = stablehlo.multiply %739, %745 : tensor<1x4096x576xf32>
    %747 = stablehlo.convert %746 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %748 = stablehlo.broadcast_in_dim %arg59, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %749 = stablehlo.multiply %747, %748 : tensor<1x4096x576xbf16>
    %750 = mhlo.copy %749 : tensor<1x4096x576xbf16>
    %751 = "stablehlo.all_gather"(%arg56) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 48, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %752 = stablehlo.dot_general %750, %751, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %753 = stablehlo.negate %752 : tensor<1x4096x2016xbf16>
    %754 = stablehlo.exponential %753 : tensor<1x4096x2016xbf16>
    %755 = stablehlo.add %754, %152 : tensor<1x4096x2016xbf16>
    %756 = stablehlo.divide %152, %755 : tensor<1x4096x2016xbf16>
    %757 = stablehlo.multiply %752, %756 : tensor<1x4096x2016xbf16>
    %758 = "stablehlo.all_gather"(%arg57) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 49, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %759 = stablehlo.dot_general %750, %758, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x2016xbf16>
    %760 = stablehlo.reduce_precision %759, format = e8m7 : tensor<1x4096x2016xbf16>
    %761 = stablehlo.multiply %757, %760 : tensor<1x4096x2016xbf16>
    %762 = mhlo.copy %761 : tensor<1x4096x2016xbf16>
    %763 = "stablehlo.all_gather"(%arg58) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 50, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %764 = stablehlo.dot_general %762, %763, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x576xbf16>
    %765 = mhlo.copy %764 : tensor<1x4096x576xbf16>
    %766 = stablehlo.add %765, %738 : tensor<1x4096x576xbf16>
    %767 = mhlo.copy %766 : tensor<1x4096x576xbf16>
    %768 = stablehlo.convert %767 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %769 = stablehlo.multiply %768, %768 : tensor<1x4096x576xf32>
    %770 = stablehlo.reduce(%769 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %771 = stablehlo.multiply %770, %34 : tensor<1x4096xf32>
    %772 = stablehlo.add %771, %36 : tensor<1x4096xf32>
    %773 = stablehlo.reshape %772 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %774 = stablehlo.rsqrt %773 : tensor<1x4096x1xf32>
    %775 = stablehlo.reshape %774 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %776 = stablehlo.broadcast_in_dim %775, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %777 = stablehlo.multiply %768, %776 : tensor<1x4096x576xf32>
    %778 = stablehlo.convert %777 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %779 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x4096xi32>
    %780 = stablehlo.compare  NE, %arg205, %779 : (tensor<1x4096xi32>, tensor<1x4096xi32>) -> tensor<1x4096xi1>
    %781 = stablehlo.convert %780 : (tensor<1x4096xi1>) -> tensor<1x4096xi32>
    %782 = stablehlo.reduce(%781 init: %c_1) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096xi32>, tensor<i32>) -> tensor<i32>
    %783 = "stablehlo.all_reduce"(%782) <{channel_handle = #stablehlo.channel_handle<handle = 51, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<i32>, %arg207: tensor<i32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<i32>
      stablehlo.return %5085 : tensor<i32>
    }) : (tensor<i32>) -> tensor<i32>
    %784 = stablehlo.convert %783 : (tensor<i32>) -> tensor<f32>
    %cst_22 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %785 = stablehlo.add %784, %cst_22 : tensor<f32>
    %786 = stablehlo.divide %cst, %785 : tensor<f32>
    %787 = stablehlo.broadcast_in_dim %786, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %788 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %789 = stablehlo.select %780, %787, %788 : tensor<1x4096xi1>, tensor<1x4096xf32>
    %790 = mhlo.copy %789 : tensor<1x4096xf32>
    %791 = stablehlo.broadcast_in_dim %790, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x128256xf32>
    %792 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %793 = stablehlo.multiply %778, %792 : tensor<1x4096x576xbf16>
    %794 = "stablehlo.all_gather"(%arg65) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 52, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x128256xbf16>) -> tensor<576x128256xbf16>
    %795 = stablehlo.dot_general %793, %794, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<576x128256xbf16>) -> tensor<1x4096x128256xbf16>
    %796 = mhlo.copy %795 : tensor<1x4096x128256xbf16>
    %797 = stablehlo.convert %796 : (tensor<1x4096x128256xbf16>) -> tensor<1x4096x128256xf32>
    %cst_23 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %798 = stablehlo.reduce(%797 init: %cst_23) applies stablehlo.maximum across dimensions = [2] : (tensor<1x4096x128256xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %799 = stablehlo.broadcast_in_dim %798, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x128256xf32>
    %800 = stablehlo.subtract %797, %799 : tensor<1x4096x128256xf32>
    %801 = stablehlo.exponential %800 : tensor<1x4096x128256xf32>
    %802 = stablehlo.reduce(%801 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x128256xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %803 = stablehlo.log %802 : tensor<1x4096xf32>
    %804 = stablehlo.add %803, %798 : tensor<1x4096xf32>
    %805 = stablehlo.multiply %804, %788 : tensor<1x4096xf32>
    %806 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %807 = stablehlo.add %805, %806 : tensor<1x4096xf32>
    %808 = stablehlo.broadcast_in_dim %807, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x128256xf32>
    %809 = stablehlo.multiply %808, %801 : tensor<1x4096x128256xf32>
    %810 = stablehlo.broadcast_in_dim %802, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x128256xf32>
    %811 = stablehlo.divide %809, %810 : tensor<1x4096x128256xf32>
    %812 = stablehlo.broadcast_in_dim %arg204, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x128256xi32>
    %813 = stablehlo.iota dim = 2 : tensor<1x4096x128256xi32>
    %814 = stablehlo.compare  EQ, %812, %813 : (tensor<1x4096x128256xi32>, tensor<1x4096x128256xi32>) -> tensor<1x4096x128256xi1>
    %815 = stablehlo.convert %814 : (tensor<1x4096x128256xi1>) -> tensor<1x4096x128256xf32>
    %816 = stablehlo.subtract %811, %815 : tensor<1x4096x128256xf32>
    %817 = stablehlo.multiply %791, %816 : tensor<1x4096x128256xf32>
    %818 = stablehlo.convert %817 : (tensor<1x4096x128256xf32>) -> tensor<1x4096x128256xbf16>
    %819 = mhlo.copy %818 : tensor<1x4096x128256xbf16>
    %820 = stablehlo.dot_general %819, %794, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x128256xbf16>, tensor<576x128256xbf16>) -> tensor<1x4096x576xbf16>
    %821 = stablehlo.multiply %778, %820 : tensor<1x4096x576xbf16>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %822 = stablehlo.reduce(%821 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %823 = "stablehlo.all_reduce"(%822) <{channel_handle = #stablehlo.channel_handle<handle = 53, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %824 = stablehlo.multiply %823, %823 : tensor<576xbf16>
    %825 = stablehlo.convert %824 : (tensor<576xbf16>) -> tensor<576xf32>
    %826 = stablehlo.reduce(%825 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %827 = stablehlo.convert %826 : (tensor<f32>) -> tensor<bf16>
    %828 = stablehlo.multiply %820, %792 : tensor<1x4096x576xbf16>
    %829 = stablehlo.convert %828 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %830 = stablehlo.multiply %829, %776 : tensor<1x4096x576xf32>
    %831 = stablehlo.multiply %768, %829 : tensor<1x4096x576xf32>
    %832 = stablehlo.reduce(%831 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %833 = stablehlo.reshape %832 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %834 = stablehlo.divide %774, %773 : tensor<1x4096x1xf32>
    %cst_25 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %835 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x4096x1xf32>
    %836 = stablehlo.multiply %834, %835 : tensor<1x4096x1xf32>
    %837 = stablehlo.multiply %833, %836 : tensor<1x4096x1xf32>
    %cst_26 = stablehlo.constant dense<0.00347222225> : tensor<f32>
    %838 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x4096x1xf32>
    %839 = stablehlo.multiply %837, %838 : tensor<1x4096x1xf32>
    %840 = stablehlo.reshape %839 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %841 = stablehlo.broadcast_in_dim %840, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %842 = stablehlo.multiply %768, %841 : tensor<1x4096x576xf32>
    %843 = stablehlo.add %830, %842 : tensor<1x4096x576xf32>
    %844 = stablehlo.convert %843 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %845:19 = stablehlo.optimization_barrier %711, %698, %685, %736, %752, %760, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %667, %arg203, %arg202, %844 : tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>, tensor<1x4096x2016xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096xi32>, tensor<1x4096xi32>, tensor<1x4096x576xbf16>
    %846 = mhlo.copy %845#18 : tensor<1x4096x576xbf16>
    %847 = stablehlo.negate %845#4 : tensor<1x4096x2016xbf16>
    %848 = stablehlo.exponential %847 : tensor<1x4096x2016xbf16>
    %849 = stablehlo.add %848, %152 : tensor<1x4096x2016xbf16>
    %850 = stablehlo.divide %152, %849 : tensor<1x4096x2016xbf16>
    %851 = stablehlo.multiply %845#4, %850 : tensor<1x4096x2016xbf16>
    %852 = mhlo.copy %846 : tensor<1x4096x576xbf16>
    %853 = "stablehlo.all_gather"(%845#8) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 54, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %854 = stablehlo.dot_general %852, %853, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x2016xbf16>
    %855 = mhlo.copy %854 : tensor<1x4096x2016xbf16>
    %856 = stablehlo.multiply %851, %855 : tensor<1x4096x2016xbf16>
    %857 = "stablehlo.all_gather"(%845#7) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 55, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %858 = stablehlo.dot_general %856, %857, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %859 = stablehlo.multiply %855, %845#5 : tensor<1x4096x2016xbf16>
    %860 = stablehlo.multiply %859, %850 : tensor<1x4096x2016xbf16>
    %861 = stablehlo.multiply %845#4, %859 : tensor<1x4096x2016xbf16>
    %862 = stablehlo.subtract %152, %850 : tensor<1x4096x2016xbf16>
    %863 = stablehlo.multiply %850, %862 : tensor<1x4096x2016xbf16>
    %864 = stablehlo.multiply %861, %863 : tensor<1x4096x2016xbf16>
    %865 = stablehlo.add %860, %864 : tensor<1x4096x2016xbf16>
    %866 = "stablehlo.all_gather"(%845#6) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 56, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %867 = stablehlo.dot_general %865, %866, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %868 = stablehlo.add %858, %867 : tensor<1x4096x576xbf16>
    %869 = mhlo.copy %868 : tensor<1x4096x576xbf16>
    %870 = stablehlo.broadcast_in_dim %845#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %871 = stablehlo.multiply %869, %870 : tensor<1x4096x576xbf16>
    %872 = stablehlo.convert %871 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %873 = mhlo.copy %845#15 : tensor<1x4096x576xbf16>
    %874 = mhlo.copy %845#3 : tensor<1x4096x576xbf16>
    %875 = stablehlo.add %873, %874 : tensor<1x4096x576xbf16>
    %876 = stablehlo.convert %875 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %877 = stablehlo.multiply %876, %876 : tensor<1x4096x576xf32>
    %878 = stablehlo.reduce(%877 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %879 = stablehlo.multiply %878, %34 : tensor<1x4096xf32>
    %880 = stablehlo.add %879, %36 : tensor<1x4096xf32>
    %881 = stablehlo.reshape %880 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %882 = stablehlo.rsqrt %881 : tensor<1x4096x1xf32>
    %883 = stablehlo.reshape %882 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %884 = stablehlo.broadcast_in_dim %883, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %885 = stablehlo.multiply %872, %884 : tensor<1x4096x576xf32>
    %886 = stablehlo.multiply %876, %872 : tensor<1x4096x576xf32>
    %887 = stablehlo.reduce(%886 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %888 = stablehlo.reshape %887 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %889 = stablehlo.divide %882, %881 : tensor<1x4096x1xf32>
    %890 = stablehlo.multiply %889, %835 : tensor<1x4096x1xf32>
    %891 = stablehlo.multiply %888, %890 : tensor<1x4096x1xf32>
    %892 = stablehlo.multiply %891, %838 : tensor<1x4096x1xf32>
    %893 = stablehlo.reshape %892 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %894 = stablehlo.broadcast_in_dim %893, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %895 = stablehlo.multiply %876, %894 : tensor<1x4096x576xf32>
    %896 = stablehlo.add %885, %895 : tensor<1x4096x576xf32>
    %897 = stablehlo.convert %896 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %898 = stablehlo.add %846, %897 : tensor<1x4096x576xbf16>
    %899 = mhlo.copy %898 : tensor<1x4096x576xbf16>
    %900 = "stablehlo.all_gather"(%845#12) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 57, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %901 = stablehlo.dot_general %899, %900, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x9x128xbf16>
    %902 = mhlo.copy %901 : tensor<1x4096x9x128xbf16>
    %903 = stablehlo.broadcast_in_dim %845#16, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %904 = stablehlo.broadcast_in_dim %845#16, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %905 = stablehlo.compare  EQ, %903, %904 : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %906 = stablehlo.and %905, %56 : tensor<1x4096x4096xi1>
    %907 = stablehlo.select %906, %58, %59 : tensor<1x4096x4096xi1>, tensor<1x4096x4096xf32>
    %908 = stablehlo.compare  GE, %907, %61 : (tensor<1x4096x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xi1>
    %909 = stablehlo.broadcast_in_dim %908, dims = [0, 3, 4] : (tensor<1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xi1>
    %910 = stablehlo.dot_general %845#1, %845#0, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %911 = stablehlo.reshape %910 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %912 = stablehlo.transpose %911, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %913 = stablehlo.select %909, %912, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %914 = stablehlo.reduce(%913 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %915 = stablehlo.broadcast_in_dim %914, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %916 = stablehlo.subtract %913, %915 : tensor<1x9x1x4096x4096xbf16>
    %917 = stablehlo.exponential %916 : tensor<1x9x1x4096x4096xbf16>
    %918 = stablehlo.reshape %917 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %919 = stablehlo.convert %918 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %920 = stablehlo.reduce(%919 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %921 = stablehlo.convert %920 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %922 = stablehlo.reshape %921 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %923 = stablehlo.transpose %922, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %924 = stablehlo.reshape %923 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %925 = stablehlo.broadcast_in_dim %924, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %926 = stablehlo.divide %902, %925 : tensor<1x4096x9x128xbf16>
    %927 = stablehlo.reshape %926 : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %928 = stablehlo.transpose %927, dims = [0, 2, 4, 3, 1] {result_layout = dense<[2, 3, 1, 4, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,128,1,4096]{2,3,1,4,0}"} : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %929 = stablehlo.reshape %928 {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,128,4096]{2,1,3,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %930 = stablehlo.dot_general %929, %918, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %931 = stablehlo.transpose %930, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{1,3,2,0}"} : (tensor<1x9x128x4096xbf16>) -> tensor<1x4096x9x128xbf16>
    %932 = mhlo.copy %931 : tensor<1x4096x9x128xbf16>
    %933 = "stablehlo.all_gather"(%845#14) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 58, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %934 = stablehlo.dot_general %932, %933, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %935 = stablehlo.broadcast_in_dim %908, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x9x4096x4096xi1>
    %936 = stablehlo.dot_general %929, %845#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %937 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<bf16>) -> tensor<1x4096x9x1xbf16>
    %938 = stablehlo.multiply %923, %923 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : tensor<1x4096x9x1xbf16>
    %939 = stablehlo.divide %937, %938 : tensor<1x4096x9x1xbf16>
    %940 = stablehlo.reshape %939 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %941 = stablehlo.broadcast_in_dim %940, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %942 = stablehlo.multiply %902, %941 : tensor<1x4096x9x128xbf16>
    %943 = stablehlo.dot_general %845#2, %918, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %944 = stablehlo.reshape %943 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %945 = stablehlo.transpose %944, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %946 = stablehlo.reshape %945 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %947 = stablehlo.multiply %942, %946 : tensor<1x4096x9x128xbf16>
    %948 = stablehlo.reduce(%947 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x4096x9x128xbf16>, tensor<bf16>) -> tensor<1x4096x9xbf16>
    %949 = stablehlo.negate %948 : tensor<1x4096x9xbf16>
    %950 = stablehlo.reshape %949 : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1xbf16>
    %951 = stablehlo.transpose %950, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,1]{3,1,2,0}"} : (tensor<1x4096x9x1xbf16>) -> tensor<1x9x4096x1xbf16>
    %952 = stablehlo.reshape %951 : (tensor<1x9x4096x1xbf16>) -> tensor<1x9x4096xbf16>
    %953 = stablehlo.broadcast_in_dim %952, dims = [0, 1, 2] : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %954 = stablehlo.add %936, %953 : tensor<1x9x4096x4096xbf16>
    %955 = stablehlo.multiply %954, %918 : tensor<1x9x4096x4096xbf16>
    %956 = stablehlo.compare  EQ, %913, %915 : (tensor<1x9x1x4096x4096xbf16>, tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xi1>
    %957 = stablehlo.reduce(%955 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %958 = stablehlo.negate %957 : tensor<1x9x4096xbf16>
    %959 = stablehlo.convert %956 : (tensor<1x9x1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xbf16>
    %960 = stablehlo.reduce(%959 init: %cst_24) applies stablehlo.add across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %961 = stablehlo.divide %958, %960 : tensor<1x9x4096xbf16>
    %962 = stablehlo.broadcast_in_dim %961, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %963 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<bf16>) -> tensor<1x9x1x4096x4096xbf16>
    %964 = stablehlo.select %956, %962, %963 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %965 = stablehlo.reshape %964 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %966 = stablehlo.add %955, %965 : tensor<1x9x4096x4096xbf16>
    %967 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<bf16>) -> tensor<1x9x4096x4096xbf16>
    %968 = stablehlo.select %935, %966, %967 : tensor<1x9x4096x4096xi1>, tensor<1x9x4096x4096xbf16>
    %969 = stablehlo.reshape %968 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %970 = stablehlo.transpose %969, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,4096,4096,1]{2,3,4,1,0}"} : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %971 = stablehlo.reshape %970 {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,4096]{2,3,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x4096x4096xbf16>
    %972 = stablehlo.dot_general %971, %845#0, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %973 = stablehlo.transpose %972, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{3,1,2,0}"} : (tensor<1x9x4096x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %974 = mhlo.copy %973 : tensor<1x4096x9x128xbf16>
    %975 = stablehlo.slice %974 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %976 = stablehlo.convert %845#17 : (tensor<1x4096xi32>) -> tensor<1x4096xf32>
    %977 = stablehlo.broadcast_in_dim %976, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x1x64xf32>
    %978 = stablehlo.divide %977, %77 : tensor<1x4096x1x64xf32>
    %979 = stablehlo.sine %978 : tensor<1x4096x1x64xf32>
    %980 = stablehlo.convert %979 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %981 = stablehlo.reshape %980 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %982 = stablehlo.broadcast_in_dim %981, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %983 = stablehlo.multiply %975, %982 : tensor<1x4096x9x64xbf16>
    %984 = stablehlo.slice %974 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %985 = stablehlo.cosine %978 : tensor<1x4096x1x64xf32>
    %986 = stablehlo.convert %985 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %987 = stablehlo.reshape %986 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %988 = stablehlo.broadcast_in_dim %987, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %989 = stablehlo.multiply %984, %988 : tensor<1x4096x9x64xbf16>
    %990 = stablehlo.add %983, %989 : tensor<1x4096x9x64xbf16>
    %991 = stablehlo.multiply %975, %988 : tensor<1x4096x9x64xbf16>
    %992 = stablehlo.negate %984 : tensor<1x4096x9x64xbf16>
    %993 = stablehlo.multiply %992, %982 : tensor<1x4096x9x64xbf16>
    %994 = stablehlo.add %991, %993 : tensor<1x4096x9x64xbf16>
    %995 = stablehlo.concatenate %990, %994, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %996 = "stablehlo.all_gather"(%845#11) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 59, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %997 = stablehlo.dot_general %995, %996, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %998 = stablehlo.add %934, %997 : tensor<1x4096x576xbf16>
    %999 = mhlo.copy %998 : tensor<1x4096x576xbf16>
    %1000 = stablehlo.dot_general %971, %845#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %1001 = stablehlo.reshape %1000 : (tensor<1x9x4096x128xbf16>) -> tensor<1x9x4096x1x128xbf16>
    %1002 = stablehlo.transpose %1001, dims = [0, 2, 1, 3, 4] {result_layout = dense<[4, 3, 1, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{4,3,1,2,0}"} : (tensor<1x9x4096x1x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1003 = stablehlo.reshape %1002 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1004 = mhlo.copy %1003 : tensor<1x4096x9x128xbf16>
    %1005 = stablehlo.slice %1004 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1006 = stablehlo.multiply %1005, %982 : tensor<1x4096x9x64xbf16>
    %1007 = stablehlo.slice %1004 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1008 = stablehlo.multiply %1007, %988 : tensor<1x4096x9x64xbf16>
    %1009 = stablehlo.add %1006, %1008 : tensor<1x4096x9x64xbf16>
    %1010 = stablehlo.multiply %1005, %988 : tensor<1x4096x9x64xbf16>
    %1011 = stablehlo.negate %1007 : tensor<1x4096x9x64xbf16>
    %1012 = stablehlo.multiply %1011, %982 : tensor<1x4096x9x64xbf16>
    %1013 = stablehlo.add %1010, %1012 : tensor<1x4096x9x64xbf16>
    %1014 = stablehlo.concatenate %1009, %1013, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %1015 = "stablehlo.all_gather"(%845#13) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 60, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1016 = stablehlo.dot_general %1014, %1015, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1017 = mhlo.copy %1016 : tensor<1x4096x576xbf16>
    %1018 = stablehlo.add %999, %1017 : tensor<1x4096x576xbf16>
    %1019 = mhlo.copy %1018 : tensor<1x4096x576xbf16>
    %1020 = stablehlo.broadcast_in_dim %845#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %1021 = stablehlo.multiply %1019, %1020 : tensor<1x4096x576xbf16>
    %1022 = stablehlo.convert %1021 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1023 = stablehlo.convert %873 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1024 = stablehlo.multiply %1023, %1023 : tensor<1x4096x576xf32>
    %1025 = stablehlo.reduce(%1024 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1026 = stablehlo.multiply %1025, %34 : tensor<1x4096xf32>
    %1027 = stablehlo.add %1026, %36 : tensor<1x4096xf32>
    %1028 = stablehlo.reshape %1027 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1029 = stablehlo.rsqrt %1028 : tensor<1x4096x1xf32>
    %1030 = stablehlo.reshape %1029 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1031 = stablehlo.broadcast_in_dim %1030, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1032 = stablehlo.multiply %1022, %1031 : tensor<1x4096x576xf32>
    %1033 = stablehlo.multiply %1023, %1022 : tensor<1x4096x576xf32>
    %1034 = stablehlo.reduce(%1033 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1035 = stablehlo.reshape %1034 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1036 = stablehlo.divide %1029, %1028 : tensor<1x4096x1xf32>
    %1037 = stablehlo.multiply %1036, %835 : tensor<1x4096x1xf32>
    %1038 = stablehlo.multiply %1035, %1037 : tensor<1x4096x1xf32>
    %1039 = stablehlo.multiply %1038, %838 : tensor<1x4096x1xf32>
    %1040 = stablehlo.reshape %1039 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1041 = stablehlo.broadcast_in_dim %1040, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1042 = stablehlo.multiply %1023, %1041 : tensor<1x4096x576xf32>
    %1043 = stablehlo.add %1032, %1042 : tensor<1x4096x576xf32>
    %1044 = stablehlo.convert %1043 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %1045 = stablehlo.add %898, %1044 : tensor<1x4096x576xbf16>
    %1046 = mhlo.copy %1045 : tensor<1x4096x576xbf16>
    %1047:19 = stablehlo.optimization_barrier %611, %598, %585, %636, %652, %660, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %567, %arg203, %arg202, %1046 : tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>, tensor<1x4096x2016xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096xi32>, tensor<1x4096xi32>, tensor<1x4096x576xbf16>
    %1048 = mhlo.copy %1047#18 : tensor<1x4096x576xbf16>
    %1049 = stablehlo.negate %1047#4 : tensor<1x4096x2016xbf16>
    %1050 = stablehlo.exponential %1049 : tensor<1x4096x2016xbf16>
    %1051 = stablehlo.add %1050, %152 : tensor<1x4096x2016xbf16>
    %1052 = stablehlo.divide %152, %1051 : tensor<1x4096x2016xbf16>
    %1053 = stablehlo.multiply %1047#4, %1052 : tensor<1x4096x2016xbf16>
    %1054 = mhlo.copy %1048 : tensor<1x4096x576xbf16>
    %1055 = "stablehlo.all_gather"(%1047#8) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 61, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %1056 = stablehlo.dot_general %1054, %1055, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x2016xbf16>
    %1057 = mhlo.copy %1056 : tensor<1x4096x2016xbf16>
    %1058 = stablehlo.multiply %1053, %1057 : tensor<1x4096x2016xbf16>
    %1059 = "stablehlo.all_gather"(%1047#7) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 62, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %1060 = stablehlo.dot_general %1058, %1059, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %1061 = stablehlo.multiply %1057, %1047#5 : tensor<1x4096x2016xbf16>
    %1062 = stablehlo.multiply %1061, %1052 : tensor<1x4096x2016xbf16>
    %1063 = stablehlo.multiply %1047#4, %1061 : tensor<1x4096x2016xbf16>
    %1064 = stablehlo.subtract %152, %1052 : tensor<1x4096x2016xbf16>
    %1065 = stablehlo.multiply %1052, %1064 : tensor<1x4096x2016xbf16>
    %1066 = stablehlo.multiply %1063, %1065 : tensor<1x4096x2016xbf16>
    %1067 = stablehlo.add %1062, %1066 : tensor<1x4096x2016xbf16>
    %1068 = "stablehlo.all_gather"(%1047#6) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 63, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %1069 = stablehlo.dot_general %1067, %1068, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %1070 = stablehlo.add %1060, %1069 : tensor<1x4096x576xbf16>
    %1071 = mhlo.copy %1070 : tensor<1x4096x576xbf16>
    %1072 = stablehlo.broadcast_in_dim %1047#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %1073 = stablehlo.multiply %1071, %1072 : tensor<1x4096x576xbf16>
    %1074 = stablehlo.convert %1073 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1075 = mhlo.copy %1047#15 : tensor<1x4096x576xbf16>
    %1076 = mhlo.copy %1047#3 : tensor<1x4096x576xbf16>
    %1077 = stablehlo.add %1075, %1076 : tensor<1x4096x576xbf16>
    %1078 = stablehlo.convert %1077 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1079 = stablehlo.multiply %1078, %1078 : tensor<1x4096x576xf32>
    %1080 = stablehlo.reduce(%1079 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1081 = stablehlo.multiply %1080, %34 : tensor<1x4096xf32>
    %1082 = stablehlo.add %1081, %36 : tensor<1x4096xf32>
    %1083 = stablehlo.reshape %1082 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1084 = stablehlo.rsqrt %1083 : tensor<1x4096x1xf32>
    %1085 = stablehlo.reshape %1084 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1086 = stablehlo.broadcast_in_dim %1085, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1087 = stablehlo.multiply %1074, %1086 : tensor<1x4096x576xf32>
    %1088 = stablehlo.multiply %1078, %1074 : tensor<1x4096x576xf32>
    %1089 = stablehlo.reduce(%1088 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1090 = stablehlo.reshape %1089 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1091 = stablehlo.divide %1084, %1083 : tensor<1x4096x1xf32>
    %1092 = stablehlo.multiply %1091, %835 : tensor<1x4096x1xf32>
    %1093 = stablehlo.multiply %1090, %1092 : tensor<1x4096x1xf32>
    %1094 = stablehlo.multiply %1093, %838 : tensor<1x4096x1xf32>
    %1095 = stablehlo.reshape %1094 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1096 = stablehlo.broadcast_in_dim %1095, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1097 = stablehlo.multiply %1078, %1096 : tensor<1x4096x576xf32>
    %1098 = stablehlo.add %1087, %1097 : tensor<1x4096x576xf32>
    %1099 = stablehlo.convert %1098 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %1100 = stablehlo.add %1048, %1099 : tensor<1x4096x576xbf16>
    %1101 = mhlo.copy %1100 : tensor<1x4096x576xbf16>
    %1102 = "stablehlo.all_gather"(%1047#12) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 64, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %1103 = stablehlo.dot_general %1101, %1102, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x9x128xbf16>
    %1104 = mhlo.copy %1103 : tensor<1x4096x9x128xbf16>
    %1105 = stablehlo.broadcast_in_dim %1047#16, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1106 = stablehlo.broadcast_in_dim %1047#16, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1107 = stablehlo.compare  EQ, %1105, %1106 : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %1108 = stablehlo.and %1107, %56 : tensor<1x4096x4096xi1>
    %1109 = stablehlo.select %1108, %58, %59 : tensor<1x4096x4096xi1>, tensor<1x4096x4096xf32>
    %1110 = stablehlo.compare  GE, %1109, %61 : (tensor<1x4096x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xi1>
    %1111 = stablehlo.broadcast_in_dim %1110, dims = [0, 3, 4] : (tensor<1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xi1>
    %1112 = stablehlo.dot_general %1047#1, %1047#0, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1113 = stablehlo.reshape %1112 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %1114 = stablehlo.transpose %1113, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1115 = stablehlo.select %1111, %1114, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %1116 = stablehlo.reduce(%1115 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1117 = stablehlo.broadcast_in_dim %1116, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1118 = stablehlo.subtract %1115, %1117 : tensor<1x9x1x4096x4096xbf16>
    %1119 = stablehlo.exponential %1118 : tensor<1x9x1x4096x4096xbf16>
    %1120 = stablehlo.reshape %1119 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1121 = stablehlo.convert %1120 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %1122 = stablehlo.reduce(%1121 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %1123 = stablehlo.convert %1122 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %1124 = stablehlo.reshape %1123 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %1125 = stablehlo.transpose %1124, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %1126 = stablehlo.reshape %1125 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %1127 = stablehlo.broadcast_in_dim %1126, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %1128 = stablehlo.divide %1104, %1127 : tensor<1x4096x9x128xbf16>
    %1129 = stablehlo.reshape %1128 : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1130 = stablehlo.transpose %1129, dims = [0, 2, 4, 3, 1] {result_layout = dense<[2, 3, 1, 4, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,128,1,4096]{2,3,1,4,0}"} : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %1131 = stablehlo.reshape %1130 {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,128,4096]{2,1,3,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1132 = stablehlo.dot_general %1131, %1120, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1133 = stablehlo.transpose %1132, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{1,3,2,0}"} : (tensor<1x9x128x4096xbf16>) -> tensor<1x4096x9x128xbf16>
    %1134 = mhlo.copy %1133 : tensor<1x4096x9x128xbf16>
    %1135 = "stablehlo.all_gather"(%1047#14) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 65, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1136 = stablehlo.dot_general %1134, %1135, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1137 = stablehlo.broadcast_in_dim %1110, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x9x4096x4096xi1>
    %1138 = stablehlo.dot_general %1131, %1047#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1139 = stablehlo.multiply %1125, %1125 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : tensor<1x4096x9x1xbf16>
    %1140 = stablehlo.divide %937, %1139 : tensor<1x4096x9x1xbf16>
    %1141 = stablehlo.reshape %1140 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %1142 = stablehlo.broadcast_in_dim %1141, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %1143 = stablehlo.multiply %1104, %1142 : tensor<1x4096x9x128xbf16>
    %1144 = stablehlo.dot_general %1047#2, %1120, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1145 = stablehlo.reshape %1144 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %1146 = stablehlo.transpose %1145, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1147 = stablehlo.reshape %1146 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1148 = stablehlo.multiply %1143, %1147 : tensor<1x4096x9x128xbf16>
    %1149 = stablehlo.reduce(%1148 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x4096x9x128xbf16>, tensor<bf16>) -> tensor<1x4096x9xbf16>
    %1150 = stablehlo.negate %1149 : tensor<1x4096x9xbf16>
    %1151 = stablehlo.reshape %1150 : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1xbf16>
    %1152 = stablehlo.transpose %1151, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,1]{3,1,2,0}"} : (tensor<1x4096x9x1xbf16>) -> tensor<1x9x4096x1xbf16>
    %1153 = stablehlo.reshape %1152 : (tensor<1x9x4096x1xbf16>) -> tensor<1x9x4096xbf16>
    %1154 = stablehlo.broadcast_in_dim %1153, dims = [0, 1, 2] : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1155 = stablehlo.add %1138, %1154 : tensor<1x9x4096x4096xbf16>
    %1156 = stablehlo.multiply %1155, %1120 : tensor<1x9x4096x4096xbf16>
    %1157 = stablehlo.compare  EQ, %1115, %1117 : (tensor<1x9x1x4096x4096xbf16>, tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xi1>
    %1158 = stablehlo.reduce(%1156 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1159 = stablehlo.negate %1158 : tensor<1x9x4096xbf16>
    %1160 = stablehlo.convert %1157 : (tensor<1x9x1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xbf16>
    %1161 = stablehlo.reduce(%1160 init: %cst_24) applies stablehlo.add across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1162 = stablehlo.divide %1159, %1161 : tensor<1x9x4096xbf16>
    %1163 = stablehlo.broadcast_in_dim %1162, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1164 = stablehlo.select %1157, %1163, %963 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %1165 = stablehlo.reshape %1164 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1166 = stablehlo.add %1156, %1165 : tensor<1x9x4096x4096xbf16>
    %1167 = stablehlo.select %1137, %1166, %967 : tensor<1x9x4096x4096xi1>, tensor<1x9x4096x4096xbf16>
    %1168 = stablehlo.reshape %1167 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1169 = stablehlo.transpose %1168, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,4096,4096,1]{2,3,4,1,0}"} : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %1170 = stablehlo.reshape %1169 {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,4096]{2,3,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1171 = stablehlo.dot_general %1170, %1047#0, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %1172 = stablehlo.transpose %1171, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{3,1,2,0}"} : (tensor<1x9x4096x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1173 = mhlo.copy %1172 : tensor<1x4096x9x128xbf16>
    %1174 = stablehlo.slice %1173 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1175 = stablehlo.convert %1047#17 : (tensor<1x4096xi32>) -> tensor<1x4096xf32>
    %1176 = stablehlo.broadcast_in_dim %1175, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x1x64xf32>
    %1177 = stablehlo.divide %1176, %77 : tensor<1x4096x1x64xf32>
    %1178 = stablehlo.sine %1177 : tensor<1x4096x1x64xf32>
    %1179 = stablehlo.convert %1178 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %1180 = stablehlo.reshape %1179 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %1181 = stablehlo.broadcast_in_dim %1180, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %1182 = stablehlo.multiply %1174, %1181 : tensor<1x4096x9x64xbf16>
    %1183 = stablehlo.slice %1173 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1184 = stablehlo.cosine %1177 : tensor<1x4096x1x64xf32>
    %1185 = stablehlo.convert %1184 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %1186 = stablehlo.reshape %1185 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %1187 = stablehlo.broadcast_in_dim %1186, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %1188 = stablehlo.multiply %1183, %1187 : tensor<1x4096x9x64xbf16>
    %1189 = stablehlo.add %1182, %1188 : tensor<1x4096x9x64xbf16>
    %1190 = stablehlo.multiply %1174, %1187 : tensor<1x4096x9x64xbf16>
    %1191 = stablehlo.negate %1183 : tensor<1x4096x9x64xbf16>
    %1192 = stablehlo.multiply %1191, %1181 : tensor<1x4096x9x64xbf16>
    %1193 = stablehlo.add %1190, %1192 : tensor<1x4096x9x64xbf16>
    %1194 = stablehlo.concatenate %1189, %1193, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %1195 = "stablehlo.all_gather"(%1047#11) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 66, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1196 = stablehlo.dot_general %1194, %1195, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1197 = stablehlo.add %1136, %1196 : tensor<1x4096x576xbf16>
    %1198 = mhlo.copy %1197 : tensor<1x4096x576xbf16>
    %1199 = stablehlo.dot_general %1170, %1047#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %1200 = stablehlo.reshape %1199 : (tensor<1x9x4096x128xbf16>) -> tensor<1x9x4096x1x128xbf16>
    %1201 = stablehlo.transpose %1200, dims = [0, 2, 1, 3, 4] {result_layout = dense<[4, 3, 1, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{4,3,1,2,0}"} : (tensor<1x9x4096x1x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1202 = stablehlo.reshape %1201 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1203 = mhlo.copy %1202 : tensor<1x4096x9x128xbf16>
    %1204 = stablehlo.slice %1203 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1205 = stablehlo.multiply %1204, %1181 : tensor<1x4096x9x64xbf16>
    %1206 = stablehlo.slice %1203 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1207 = stablehlo.multiply %1206, %1187 : tensor<1x4096x9x64xbf16>
    %1208 = stablehlo.add %1205, %1207 : tensor<1x4096x9x64xbf16>
    %1209 = stablehlo.multiply %1204, %1187 : tensor<1x4096x9x64xbf16>
    %1210 = stablehlo.negate %1206 : tensor<1x4096x9x64xbf16>
    %1211 = stablehlo.multiply %1210, %1181 : tensor<1x4096x9x64xbf16>
    %1212 = stablehlo.add %1209, %1211 : tensor<1x4096x9x64xbf16>
    %1213 = stablehlo.concatenate %1208, %1212, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %1214 = "stablehlo.all_gather"(%1047#13) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 67, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1215 = stablehlo.dot_general %1213, %1214, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1216 = mhlo.copy %1215 : tensor<1x4096x576xbf16>
    %1217 = stablehlo.add %1198, %1216 : tensor<1x4096x576xbf16>
    %1218 = mhlo.copy %1217 : tensor<1x4096x576xbf16>
    %1219 = stablehlo.broadcast_in_dim %1047#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %1220 = stablehlo.multiply %1218, %1219 : tensor<1x4096x576xbf16>
    %1221 = stablehlo.convert %1220 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1222 = stablehlo.convert %1075 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1223 = stablehlo.multiply %1222, %1222 : tensor<1x4096x576xf32>
    %1224 = stablehlo.reduce(%1223 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1225 = stablehlo.multiply %1224, %34 : tensor<1x4096xf32>
    %1226 = stablehlo.add %1225, %36 : tensor<1x4096xf32>
    %1227 = stablehlo.reshape %1226 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1228 = stablehlo.rsqrt %1227 : tensor<1x4096x1xf32>
    %1229 = stablehlo.reshape %1228 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1230 = stablehlo.broadcast_in_dim %1229, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1231 = stablehlo.multiply %1221, %1230 : tensor<1x4096x576xf32>
    %1232 = stablehlo.multiply %1222, %1221 : tensor<1x4096x576xf32>
    %1233 = stablehlo.reduce(%1232 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1234 = stablehlo.reshape %1233 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1235 = stablehlo.divide %1228, %1227 : tensor<1x4096x1xf32>
    %1236 = stablehlo.multiply %1235, %835 : tensor<1x4096x1xf32>
    %1237 = stablehlo.multiply %1234, %1236 : tensor<1x4096x1xf32>
    %1238 = stablehlo.multiply %1237, %838 : tensor<1x4096x1xf32>
    %1239 = stablehlo.reshape %1238 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1240 = stablehlo.broadcast_in_dim %1239, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1241 = stablehlo.multiply %1222, %1240 : tensor<1x4096x576xf32>
    %1242 = stablehlo.add %1231, %1241 : tensor<1x4096x576xf32>
    %1243 = stablehlo.convert %1242 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %1244 = stablehlo.add %1100, %1243 : tensor<1x4096x576xbf16>
    %1245 = mhlo.copy %1244 : tensor<1x4096x576xbf16>
    %1246:19 = stablehlo.optimization_barrier %511, %498, %485, %536, %552, %560, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %467, %arg203, %arg202, %1245 : tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>, tensor<1x4096x2016xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096xi32>, tensor<1x4096xi32>, tensor<1x4096x576xbf16>
    %1247 = mhlo.copy %1246#18 : tensor<1x4096x576xbf16>
    %1248 = stablehlo.negate %1246#4 : tensor<1x4096x2016xbf16>
    %1249 = stablehlo.exponential %1248 : tensor<1x4096x2016xbf16>
    %1250 = stablehlo.add %1249, %152 : tensor<1x4096x2016xbf16>
    %1251 = stablehlo.divide %152, %1250 : tensor<1x4096x2016xbf16>
    %1252 = stablehlo.multiply %1246#4, %1251 : tensor<1x4096x2016xbf16>
    %1253 = mhlo.copy %1247 : tensor<1x4096x576xbf16>
    %1254 = "stablehlo.all_gather"(%1246#8) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 68, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %1255 = stablehlo.dot_general %1253, %1254, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x2016xbf16>
    %1256 = mhlo.copy %1255 : tensor<1x4096x2016xbf16>
    %1257 = stablehlo.multiply %1252, %1256 : tensor<1x4096x2016xbf16>
    %1258 = "stablehlo.all_gather"(%1246#7) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 69, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %1259 = stablehlo.dot_general %1257, %1258, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %1260 = stablehlo.multiply %1256, %1246#5 : tensor<1x4096x2016xbf16>
    %1261 = stablehlo.multiply %1260, %1251 : tensor<1x4096x2016xbf16>
    %1262 = stablehlo.multiply %1246#4, %1260 : tensor<1x4096x2016xbf16>
    %1263 = stablehlo.subtract %152, %1251 : tensor<1x4096x2016xbf16>
    %1264 = stablehlo.multiply %1251, %1263 : tensor<1x4096x2016xbf16>
    %1265 = stablehlo.multiply %1262, %1264 : tensor<1x4096x2016xbf16>
    %1266 = stablehlo.add %1261, %1265 : tensor<1x4096x2016xbf16>
    %1267 = "stablehlo.all_gather"(%1246#6) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 70, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %1268 = stablehlo.dot_general %1266, %1267, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %1269 = stablehlo.add %1259, %1268 : tensor<1x4096x576xbf16>
    %1270 = mhlo.copy %1269 : tensor<1x4096x576xbf16>
    %1271 = stablehlo.broadcast_in_dim %1246#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %1272 = stablehlo.multiply %1270, %1271 : tensor<1x4096x576xbf16>
    %1273 = stablehlo.convert %1272 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1274 = mhlo.copy %1246#15 : tensor<1x4096x576xbf16>
    %1275 = mhlo.copy %1246#3 : tensor<1x4096x576xbf16>
    %1276 = stablehlo.add %1274, %1275 : tensor<1x4096x576xbf16>
    %1277 = stablehlo.convert %1276 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1278 = stablehlo.multiply %1277, %1277 : tensor<1x4096x576xf32>
    %1279 = stablehlo.reduce(%1278 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1280 = stablehlo.multiply %1279, %34 : tensor<1x4096xf32>
    %1281 = stablehlo.add %1280, %36 : tensor<1x4096xf32>
    %1282 = stablehlo.reshape %1281 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1283 = stablehlo.rsqrt %1282 : tensor<1x4096x1xf32>
    %1284 = stablehlo.reshape %1283 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1285 = stablehlo.broadcast_in_dim %1284, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1286 = stablehlo.multiply %1273, %1285 : tensor<1x4096x576xf32>
    %1287 = stablehlo.multiply %1277, %1273 : tensor<1x4096x576xf32>
    %1288 = stablehlo.reduce(%1287 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1289 = stablehlo.reshape %1288 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1290 = stablehlo.divide %1283, %1282 : tensor<1x4096x1xf32>
    %1291 = stablehlo.multiply %1290, %835 : tensor<1x4096x1xf32>
    %1292 = stablehlo.multiply %1289, %1291 : tensor<1x4096x1xf32>
    %1293 = stablehlo.multiply %1292, %838 : tensor<1x4096x1xf32>
    %1294 = stablehlo.reshape %1293 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1295 = stablehlo.broadcast_in_dim %1294, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1296 = stablehlo.multiply %1277, %1295 : tensor<1x4096x576xf32>
    %1297 = stablehlo.add %1286, %1296 : tensor<1x4096x576xf32>
    %1298 = stablehlo.convert %1297 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %1299 = stablehlo.add %1247, %1298 : tensor<1x4096x576xbf16>
    %1300 = mhlo.copy %1299 : tensor<1x4096x576xbf16>
    %1301 = "stablehlo.all_gather"(%1246#12) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 71, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %1302 = stablehlo.dot_general %1300, %1301, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x9x128xbf16>
    %1303 = mhlo.copy %1302 : tensor<1x4096x9x128xbf16>
    %1304 = stablehlo.broadcast_in_dim %1246#16, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1305 = stablehlo.broadcast_in_dim %1246#16, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1306 = stablehlo.compare  EQ, %1304, %1305 : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %1307 = stablehlo.and %1306, %56 : tensor<1x4096x4096xi1>
    %1308 = stablehlo.select %1307, %58, %59 : tensor<1x4096x4096xi1>, tensor<1x4096x4096xf32>
    %1309 = stablehlo.compare  GE, %1308, %61 : (tensor<1x4096x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xi1>
    %1310 = stablehlo.broadcast_in_dim %1309, dims = [0, 3, 4] : (tensor<1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xi1>
    %1311 = stablehlo.dot_general %1246#1, %1246#0, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1312 = stablehlo.reshape %1311 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %1313 = stablehlo.transpose %1312, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1314 = stablehlo.select %1310, %1313, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %1315 = stablehlo.reduce(%1314 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1316 = stablehlo.broadcast_in_dim %1315, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1317 = stablehlo.subtract %1314, %1316 : tensor<1x9x1x4096x4096xbf16>
    %1318 = stablehlo.exponential %1317 : tensor<1x9x1x4096x4096xbf16>
    %1319 = stablehlo.reshape %1318 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1320 = stablehlo.convert %1319 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %1321 = stablehlo.reduce(%1320 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %1322 = stablehlo.convert %1321 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %1323 = stablehlo.reshape %1322 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %1324 = stablehlo.transpose %1323, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %1325 = stablehlo.reshape %1324 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %1326 = stablehlo.broadcast_in_dim %1325, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %1327 = stablehlo.divide %1303, %1326 : tensor<1x4096x9x128xbf16>
    %1328 = stablehlo.reshape %1327 : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1329 = stablehlo.transpose %1328, dims = [0, 2, 4, 3, 1] {result_layout = dense<[2, 3, 1, 4, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,128,1,4096]{2,3,1,4,0}"} : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %1330 = stablehlo.reshape %1329 {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,128,4096]{2,1,3,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1331 = stablehlo.dot_general %1330, %1319, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1332 = stablehlo.transpose %1331, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{1,3,2,0}"} : (tensor<1x9x128x4096xbf16>) -> tensor<1x4096x9x128xbf16>
    %1333 = mhlo.copy %1332 : tensor<1x4096x9x128xbf16>
    %1334 = "stablehlo.all_gather"(%1246#14) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 72, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1335 = stablehlo.dot_general %1333, %1334, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1336 = stablehlo.broadcast_in_dim %1309, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x9x4096x4096xi1>
    %1337 = stablehlo.dot_general %1330, %1246#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1338 = stablehlo.multiply %1324, %1324 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : tensor<1x4096x9x1xbf16>
    %1339 = stablehlo.divide %937, %1338 : tensor<1x4096x9x1xbf16>
    %1340 = stablehlo.reshape %1339 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %1341 = stablehlo.broadcast_in_dim %1340, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %1342 = stablehlo.multiply %1303, %1341 : tensor<1x4096x9x128xbf16>
    %1343 = stablehlo.dot_general %1246#2, %1319, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1344 = stablehlo.reshape %1343 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %1345 = stablehlo.transpose %1344, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1346 = stablehlo.reshape %1345 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1347 = stablehlo.multiply %1342, %1346 : tensor<1x4096x9x128xbf16>
    %1348 = stablehlo.reduce(%1347 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x4096x9x128xbf16>, tensor<bf16>) -> tensor<1x4096x9xbf16>
    %1349 = stablehlo.negate %1348 : tensor<1x4096x9xbf16>
    %1350 = stablehlo.reshape %1349 : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1xbf16>
    %1351 = stablehlo.transpose %1350, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,1]{3,1,2,0}"} : (tensor<1x4096x9x1xbf16>) -> tensor<1x9x4096x1xbf16>
    %1352 = stablehlo.reshape %1351 : (tensor<1x9x4096x1xbf16>) -> tensor<1x9x4096xbf16>
    %1353 = stablehlo.broadcast_in_dim %1352, dims = [0, 1, 2] : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1354 = stablehlo.add %1337, %1353 : tensor<1x9x4096x4096xbf16>
    %1355 = stablehlo.multiply %1354, %1319 : tensor<1x9x4096x4096xbf16>
    %1356 = stablehlo.compare  EQ, %1314, %1316 : (tensor<1x9x1x4096x4096xbf16>, tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xi1>
    %1357 = stablehlo.reduce(%1355 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1358 = stablehlo.negate %1357 : tensor<1x9x4096xbf16>
    %1359 = stablehlo.convert %1356 : (tensor<1x9x1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xbf16>
    %1360 = stablehlo.reduce(%1359 init: %cst_24) applies stablehlo.add across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1361 = stablehlo.divide %1358, %1360 : tensor<1x9x4096xbf16>
    %1362 = stablehlo.broadcast_in_dim %1361, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1363 = stablehlo.select %1356, %1362, %963 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %1364 = stablehlo.reshape %1363 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1365 = stablehlo.add %1355, %1364 : tensor<1x9x4096x4096xbf16>
    %1366 = stablehlo.select %1336, %1365, %967 : tensor<1x9x4096x4096xi1>, tensor<1x9x4096x4096xbf16>
    %1367 = stablehlo.reshape %1366 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1368 = stablehlo.transpose %1367, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,4096,4096,1]{2,3,4,1,0}"} : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %1369 = stablehlo.reshape %1368 {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,4096]{2,3,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1370 = stablehlo.dot_general %1369, %1246#0, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %1371 = stablehlo.transpose %1370, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{3,1,2,0}"} : (tensor<1x9x4096x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1372 = mhlo.copy %1371 : tensor<1x4096x9x128xbf16>
    %1373 = stablehlo.slice %1372 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1374 = stablehlo.convert %1246#17 : (tensor<1x4096xi32>) -> tensor<1x4096xf32>
    %1375 = stablehlo.broadcast_in_dim %1374, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x1x64xf32>
    %1376 = stablehlo.divide %1375, %77 : tensor<1x4096x1x64xf32>
    %1377 = stablehlo.sine %1376 : tensor<1x4096x1x64xf32>
    %1378 = stablehlo.convert %1377 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %1379 = stablehlo.reshape %1378 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %1380 = stablehlo.broadcast_in_dim %1379, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %1381 = stablehlo.multiply %1373, %1380 : tensor<1x4096x9x64xbf16>
    %1382 = stablehlo.slice %1372 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1383 = stablehlo.cosine %1376 : tensor<1x4096x1x64xf32>
    %1384 = stablehlo.convert %1383 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %1385 = stablehlo.reshape %1384 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %1386 = stablehlo.broadcast_in_dim %1385, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %1387 = stablehlo.multiply %1382, %1386 : tensor<1x4096x9x64xbf16>
    %1388 = stablehlo.add %1381, %1387 : tensor<1x4096x9x64xbf16>
    %1389 = stablehlo.multiply %1373, %1386 : tensor<1x4096x9x64xbf16>
    %1390 = stablehlo.negate %1382 : tensor<1x4096x9x64xbf16>
    %1391 = stablehlo.multiply %1390, %1380 : tensor<1x4096x9x64xbf16>
    %1392 = stablehlo.add %1389, %1391 : tensor<1x4096x9x64xbf16>
    %1393 = stablehlo.concatenate %1388, %1392, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %1394 = "stablehlo.all_gather"(%1246#11) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 73, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1395 = stablehlo.dot_general %1393, %1394, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1396 = stablehlo.add %1335, %1395 : tensor<1x4096x576xbf16>
    %1397 = mhlo.copy %1396 : tensor<1x4096x576xbf16>
    %1398 = stablehlo.dot_general %1369, %1246#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %1399 = stablehlo.reshape %1398 : (tensor<1x9x4096x128xbf16>) -> tensor<1x9x4096x1x128xbf16>
    %1400 = stablehlo.transpose %1399, dims = [0, 2, 1, 3, 4] {result_layout = dense<[4, 3, 1, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{4,3,1,2,0}"} : (tensor<1x9x4096x1x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1401 = stablehlo.reshape %1400 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1402 = mhlo.copy %1401 : tensor<1x4096x9x128xbf16>
    %1403 = stablehlo.slice %1402 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1404 = stablehlo.multiply %1403, %1380 : tensor<1x4096x9x64xbf16>
    %1405 = stablehlo.slice %1402 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1406 = stablehlo.multiply %1405, %1386 : tensor<1x4096x9x64xbf16>
    %1407 = stablehlo.add %1404, %1406 : tensor<1x4096x9x64xbf16>
    %1408 = stablehlo.multiply %1403, %1386 : tensor<1x4096x9x64xbf16>
    %1409 = stablehlo.negate %1405 : tensor<1x4096x9x64xbf16>
    %1410 = stablehlo.multiply %1409, %1380 : tensor<1x4096x9x64xbf16>
    %1411 = stablehlo.add %1408, %1410 : tensor<1x4096x9x64xbf16>
    %1412 = stablehlo.concatenate %1407, %1411, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %1413 = "stablehlo.all_gather"(%1246#13) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 74, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1414 = stablehlo.dot_general %1412, %1413, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1415 = mhlo.copy %1414 : tensor<1x4096x576xbf16>
    %1416 = stablehlo.add %1397, %1415 : tensor<1x4096x576xbf16>
    %1417 = mhlo.copy %1416 : tensor<1x4096x576xbf16>
    %1418 = stablehlo.broadcast_in_dim %1246#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %1419 = stablehlo.multiply %1417, %1418 : tensor<1x4096x576xbf16>
    %1420 = stablehlo.convert %1419 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1421 = stablehlo.convert %1274 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1422 = stablehlo.multiply %1421, %1421 : tensor<1x4096x576xf32>
    %1423 = stablehlo.reduce(%1422 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1424 = stablehlo.multiply %1423, %34 : tensor<1x4096xf32>
    %1425 = stablehlo.add %1424, %36 : tensor<1x4096xf32>
    %1426 = stablehlo.reshape %1425 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1427 = stablehlo.rsqrt %1426 : tensor<1x4096x1xf32>
    %1428 = stablehlo.reshape %1427 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1429 = stablehlo.broadcast_in_dim %1428, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1430 = stablehlo.multiply %1420, %1429 : tensor<1x4096x576xf32>
    %1431 = stablehlo.multiply %1421, %1420 : tensor<1x4096x576xf32>
    %1432 = stablehlo.reduce(%1431 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1433 = stablehlo.reshape %1432 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1434 = stablehlo.divide %1427, %1426 : tensor<1x4096x1xf32>
    %1435 = stablehlo.multiply %1434, %835 : tensor<1x4096x1xf32>
    %1436 = stablehlo.multiply %1433, %1435 : tensor<1x4096x1xf32>
    %1437 = stablehlo.multiply %1436, %838 : tensor<1x4096x1xf32>
    %1438 = stablehlo.reshape %1437 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1439 = stablehlo.broadcast_in_dim %1438, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1440 = stablehlo.multiply %1421, %1439 : tensor<1x4096x576xf32>
    %1441 = stablehlo.add %1430, %1440 : tensor<1x4096x576xf32>
    %1442 = stablehlo.convert %1441 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %1443 = stablehlo.add %1299, %1442 : tensor<1x4096x576xbf16>
    %1444 = mhlo.copy %1443 : tensor<1x4096x576xbf16>
    %1445:19 = stablehlo.optimization_barrier %411, %398, %385, %436, %452, %460, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %367, %arg203, %arg202, %1444 : tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>, tensor<1x4096x2016xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096xi32>, tensor<1x4096xi32>, tensor<1x4096x576xbf16>
    %1446 = mhlo.copy %1445#18 : tensor<1x4096x576xbf16>
    %1447 = stablehlo.negate %1445#4 : tensor<1x4096x2016xbf16>
    %1448 = stablehlo.exponential %1447 : tensor<1x4096x2016xbf16>
    %1449 = stablehlo.add %1448, %152 : tensor<1x4096x2016xbf16>
    %1450 = stablehlo.divide %152, %1449 : tensor<1x4096x2016xbf16>
    %1451 = stablehlo.multiply %1445#4, %1450 : tensor<1x4096x2016xbf16>
    %1452 = mhlo.copy %1446 : tensor<1x4096x576xbf16>
    %1453 = "stablehlo.all_gather"(%1445#8) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 75, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %1454 = stablehlo.dot_general %1452, %1453, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x2016xbf16>
    %1455 = mhlo.copy %1454 : tensor<1x4096x2016xbf16>
    %1456 = stablehlo.multiply %1451, %1455 : tensor<1x4096x2016xbf16>
    %1457 = "stablehlo.all_gather"(%1445#7) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 76, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %1458 = stablehlo.dot_general %1456, %1457, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %1459 = stablehlo.multiply %1455, %1445#5 : tensor<1x4096x2016xbf16>
    %1460 = stablehlo.multiply %1459, %1450 : tensor<1x4096x2016xbf16>
    %1461 = stablehlo.multiply %1445#4, %1459 : tensor<1x4096x2016xbf16>
    %1462 = stablehlo.subtract %152, %1450 : tensor<1x4096x2016xbf16>
    %1463 = stablehlo.multiply %1450, %1462 : tensor<1x4096x2016xbf16>
    %1464 = stablehlo.multiply %1461, %1463 : tensor<1x4096x2016xbf16>
    %1465 = stablehlo.add %1460, %1464 : tensor<1x4096x2016xbf16>
    %1466 = "stablehlo.all_gather"(%1445#6) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 77, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %1467 = stablehlo.dot_general %1465, %1466, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %1468 = stablehlo.add %1458, %1467 : tensor<1x4096x576xbf16>
    %1469 = mhlo.copy %1468 : tensor<1x4096x576xbf16>
    %1470 = stablehlo.broadcast_in_dim %1445#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %1471 = stablehlo.multiply %1469, %1470 : tensor<1x4096x576xbf16>
    %1472 = stablehlo.convert %1471 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1473 = mhlo.copy %1445#15 : tensor<1x4096x576xbf16>
    %1474 = mhlo.copy %1445#3 : tensor<1x4096x576xbf16>
    %1475 = stablehlo.add %1473, %1474 : tensor<1x4096x576xbf16>
    %1476 = stablehlo.convert %1475 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1477 = stablehlo.multiply %1476, %1476 : tensor<1x4096x576xf32>
    %1478 = stablehlo.reduce(%1477 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1479 = stablehlo.multiply %1478, %34 : tensor<1x4096xf32>
    %1480 = stablehlo.add %1479, %36 : tensor<1x4096xf32>
    %1481 = stablehlo.reshape %1480 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1482 = stablehlo.rsqrt %1481 : tensor<1x4096x1xf32>
    %1483 = stablehlo.reshape %1482 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1484 = stablehlo.broadcast_in_dim %1483, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1485 = stablehlo.multiply %1472, %1484 : tensor<1x4096x576xf32>
    %1486 = stablehlo.multiply %1476, %1472 : tensor<1x4096x576xf32>
    %1487 = stablehlo.reduce(%1486 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1488 = stablehlo.reshape %1487 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1489 = stablehlo.divide %1482, %1481 : tensor<1x4096x1xf32>
    %1490 = stablehlo.multiply %1489, %835 : tensor<1x4096x1xf32>
    %1491 = stablehlo.multiply %1488, %1490 : tensor<1x4096x1xf32>
    %1492 = stablehlo.multiply %1491, %838 : tensor<1x4096x1xf32>
    %1493 = stablehlo.reshape %1492 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1494 = stablehlo.broadcast_in_dim %1493, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1495 = stablehlo.multiply %1476, %1494 : tensor<1x4096x576xf32>
    %1496 = stablehlo.add %1485, %1495 : tensor<1x4096x576xf32>
    %1497 = stablehlo.convert %1496 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %1498 = stablehlo.add %1446, %1497 : tensor<1x4096x576xbf16>
    %1499 = mhlo.copy %1498 : tensor<1x4096x576xbf16>
    %1500 = "stablehlo.all_gather"(%1445#12) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 78, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %1501 = stablehlo.dot_general %1499, %1500, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x9x128xbf16>
    %1502 = mhlo.copy %1501 : tensor<1x4096x9x128xbf16>
    %1503 = stablehlo.broadcast_in_dim %1445#16, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1504 = stablehlo.broadcast_in_dim %1445#16, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1505 = stablehlo.compare  EQ, %1503, %1504 : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %1506 = stablehlo.and %1505, %56 : tensor<1x4096x4096xi1>
    %1507 = stablehlo.select %1506, %58, %59 : tensor<1x4096x4096xi1>, tensor<1x4096x4096xf32>
    %1508 = stablehlo.compare  GE, %1507, %61 : (tensor<1x4096x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xi1>
    %1509 = stablehlo.broadcast_in_dim %1508, dims = [0, 3, 4] : (tensor<1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xi1>
    %1510 = stablehlo.dot_general %1445#1, %1445#0, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1511 = stablehlo.reshape %1510 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %1512 = stablehlo.transpose %1511, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1513 = stablehlo.select %1509, %1512, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %1514 = stablehlo.reduce(%1513 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1515 = stablehlo.broadcast_in_dim %1514, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1516 = stablehlo.subtract %1513, %1515 : tensor<1x9x1x4096x4096xbf16>
    %1517 = stablehlo.exponential %1516 : tensor<1x9x1x4096x4096xbf16>
    %1518 = stablehlo.reshape %1517 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1519 = stablehlo.convert %1518 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %1520 = stablehlo.reduce(%1519 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %1521 = stablehlo.convert %1520 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %1522 = stablehlo.reshape %1521 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %1523 = stablehlo.transpose %1522, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %1524 = stablehlo.reshape %1523 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %1525 = stablehlo.broadcast_in_dim %1524, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %1526 = stablehlo.divide %1502, %1525 : tensor<1x4096x9x128xbf16>
    %1527 = stablehlo.reshape %1526 : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1528 = stablehlo.transpose %1527, dims = [0, 2, 4, 3, 1] {result_layout = dense<[2, 3, 1, 4, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,128,1,4096]{2,3,1,4,0}"} : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %1529 = stablehlo.reshape %1528 {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,128,4096]{2,1,3,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1530 = stablehlo.dot_general %1529, %1518, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1531 = stablehlo.transpose %1530, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{1,3,2,0}"} : (tensor<1x9x128x4096xbf16>) -> tensor<1x4096x9x128xbf16>
    %1532 = mhlo.copy %1531 : tensor<1x4096x9x128xbf16>
    %1533 = "stablehlo.all_gather"(%1445#14) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 79, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1534 = stablehlo.dot_general %1532, %1533, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1535 = stablehlo.broadcast_in_dim %1508, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x9x4096x4096xi1>
    %1536 = stablehlo.dot_general %1529, %1445#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1537 = stablehlo.multiply %1523, %1523 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : tensor<1x4096x9x1xbf16>
    %1538 = stablehlo.divide %937, %1537 : tensor<1x4096x9x1xbf16>
    %1539 = stablehlo.reshape %1538 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %1540 = stablehlo.broadcast_in_dim %1539, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %1541 = stablehlo.multiply %1502, %1540 : tensor<1x4096x9x128xbf16>
    %1542 = stablehlo.dot_general %1445#2, %1518, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1543 = stablehlo.reshape %1542 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %1544 = stablehlo.transpose %1543, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1545 = stablehlo.reshape %1544 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1546 = stablehlo.multiply %1541, %1545 : tensor<1x4096x9x128xbf16>
    %1547 = stablehlo.reduce(%1546 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x4096x9x128xbf16>, tensor<bf16>) -> tensor<1x4096x9xbf16>
    %1548 = stablehlo.negate %1547 : tensor<1x4096x9xbf16>
    %1549 = stablehlo.reshape %1548 : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1xbf16>
    %1550 = stablehlo.transpose %1549, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,1]{3,1,2,0}"} : (tensor<1x4096x9x1xbf16>) -> tensor<1x9x4096x1xbf16>
    %1551 = stablehlo.reshape %1550 : (tensor<1x9x4096x1xbf16>) -> tensor<1x9x4096xbf16>
    %1552 = stablehlo.broadcast_in_dim %1551, dims = [0, 1, 2] : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1553 = stablehlo.add %1536, %1552 : tensor<1x9x4096x4096xbf16>
    %1554 = stablehlo.multiply %1553, %1518 : tensor<1x9x4096x4096xbf16>
    %1555 = stablehlo.compare  EQ, %1513, %1515 : (tensor<1x9x1x4096x4096xbf16>, tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xi1>
    %1556 = stablehlo.reduce(%1554 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1557 = stablehlo.negate %1556 : tensor<1x9x4096xbf16>
    %1558 = stablehlo.convert %1555 : (tensor<1x9x1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xbf16>
    %1559 = stablehlo.reduce(%1558 init: %cst_24) applies stablehlo.add across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1560 = stablehlo.divide %1557, %1559 : tensor<1x9x4096xbf16>
    %1561 = stablehlo.broadcast_in_dim %1560, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1562 = stablehlo.select %1555, %1561, %963 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %1563 = stablehlo.reshape %1562 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1564 = stablehlo.add %1554, %1563 : tensor<1x9x4096x4096xbf16>
    %1565 = stablehlo.select %1535, %1564, %967 : tensor<1x9x4096x4096xi1>, tensor<1x9x4096x4096xbf16>
    %1566 = stablehlo.reshape %1565 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1567 = stablehlo.transpose %1566, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,4096,4096,1]{2,3,4,1,0}"} : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %1568 = stablehlo.reshape %1567 {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,4096]{2,3,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1569 = stablehlo.dot_general %1568, %1445#0, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %1570 = stablehlo.transpose %1569, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{3,1,2,0}"} : (tensor<1x9x4096x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1571 = mhlo.copy %1570 : tensor<1x4096x9x128xbf16>
    %1572 = stablehlo.slice %1571 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1573 = stablehlo.convert %1445#17 : (tensor<1x4096xi32>) -> tensor<1x4096xf32>
    %1574 = stablehlo.broadcast_in_dim %1573, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x1x64xf32>
    %1575 = stablehlo.divide %1574, %77 : tensor<1x4096x1x64xf32>
    %1576 = stablehlo.sine %1575 : tensor<1x4096x1x64xf32>
    %1577 = stablehlo.convert %1576 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %1578 = stablehlo.reshape %1577 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %1579 = stablehlo.broadcast_in_dim %1578, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %1580 = stablehlo.multiply %1572, %1579 : tensor<1x4096x9x64xbf16>
    %1581 = stablehlo.slice %1571 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1582 = stablehlo.cosine %1575 : tensor<1x4096x1x64xf32>
    %1583 = stablehlo.convert %1582 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %1584 = stablehlo.reshape %1583 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %1585 = stablehlo.broadcast_in_dim %1584, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %1586 = stablehlo.multiply %1581, %1585 : tensor<1x4096x9x64xbf16>
    %1587 = stablehlo.add %1580, %1586 : tensor<1x4096x9x64xbf16>
    %1588 = stablehlo.multiply %1572, %1585 : tensor<1x4096x9x64xbf16>
    %1589 = stablehlo.negate %1581 : tensor<1x4096x9x64xbf16>
    %1590 = stablehlo.multiply %1589, %1579 : tensor<1x4096x9x64xbf16>
    %1591 = stablehlo.add %1588, %1590 : tensor<1x4096x9x64xbf16>
    %1592 = stablehlo.concatenate %1587, %1591, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %1593 = "stablehlo.all_gather"(%1445#11) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 80, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1594 = stablehlo.dot_general %1592, %1593, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1595 = stablehlo.add %1534, %1594 : tensor<1x4096x576xbf16>
    %1596 = mhlo.copy %1595 : tensor<1x4096x576xbf16>
    %1597 = stablehlo.dot_general %1568, %1445#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %1598 = stablehlo.reshape %1597 : (tensor<1x9x4096x128xbf16>) -> tensor<1x9x4096x1x128xbf16>
    %1599 = stablehlo.transpose %1598, dims = [0, 2, 1, 3, 4] {result_layout = dense<[4, 3, 1, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{4,3,1,2,0}"} : (tensor<1x9x4096x1x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1600 = stablehlo.reshape %1599 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1601 = mhlo.copy %1600 : tensor<1x4096x9x128xbf16>
    %1602 = stablehlo.slice %1601 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1603 = stablehlo.multiply %1602, %1579 : tensor<1x4096x9x64xbf16>
    %1604 = stablehlo.slice %1601 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1605 = stablehlo.multiply %1604, %1585 : tensor<1x4096x9x64xbf16>
    %1606 = stablehlo.add %1603, %1605 : tensor<1x4096x9x64xbf16>
    %1607 = stablehlo.multiply %1602, %1585 : tensor<1x4096x9x64xbf16>
    %1608 = stablehlo.negate %1604 : tensor<1x4096x9x64xbf16>
    %1609 = stablehlo.multiply %1608, %1579 : tensor<1x4096x9x64xbf16>
    %1610 = stablehlo.add %1607, %1609 : tensor<1x4096x9x64xbf16>
    %1611 = stablehlo.concatenate %1606, %1610, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %1612 = "stablehlo.all_gather"(%1445#13) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 81, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1613 = stablehlo.dot_general %1611, %1612, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1614 = mhlo.copy %1613 : tensor<1x4096x576xbf16>
    %1615 = stablehlo.add %1596, %1614 : tensor<1x4096x576xbf16>
    %1616 = mhlo.copy %1615 : tensor<1x4096x576xbf16>
    %1617 = stablehlo.broadcast_in_dim %1445#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %1618 = stablehlo.multiply %1616, %1617 : tensor<1x4096x576xbf16>
    %1619 = stablehlo.convert %1618 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1620 = stablehlo.convert %1473 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1621 = stablehlo.multiply %1620, %1620 : tensor<1x4096x576xf32>
    %1622 = stablehlo.reduce(%1621 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1623 = stablehlo.multiply %1622, %34 : tensor<1x4096xf32>
    %1624 = stablehlo.add %1623, %36 : tensor<1x4096xf32>
    %1625 = stablehlo.reshape %1624 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1626 = stablehlo.rsqrt %1625 : tensor<1x4096x1xf32>
    %1627 = stablehlo.reshape %1626 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1628 = stablehlo.broadcast_in_dim %1627, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1629 = stablehlo.multiply %1619, %1628 : tensor<1x4096x576xf32>
    %1630 = stablehlo.multiply %1620, %1619 : tensor<1x4096x576xf32>
    %1631 = stablehlo.reduce(%1630 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1632 = stablehlo.reshape %1631 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1633 = stablehlo.divide %1626, %1625 : tensor<1x4096x1xf32>
    %1634 = stablehlo.multiply %1633, %835 : tensor<1x4096x1xf32>
    %1635 = stablehlo.multiply %1632, %1634 : tensor<1x4096x1xf32>
    %1636 = stablehlo.multiply %1635, %838 : tensor<1x4096x1xf32>
    %1637 = stablehlo.reshape %1636 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1638 = stablehlo.broadcast_in_dim %1637, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1639 = stablehlo.multiply %1620, %1638 : tensor<1x4096x576xf32>
    %1640 = stablehlo.add %1629, %1639 : tensor<1x4096x576xf32>
    %1641 = stablehlo.convert %1640 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %1642 = stablehlo.add %1498, %1641 : tensor<1x4096x576xbf16>
    %1643 = mhlo.copy %1642 : tensor<1x4096x576xbf16>
    %1644:19 = stablehlo.optimization_barrier %311, %298, %285, %336, %352, %360, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %267, %arg203, %arg202, %1643 : tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>, tensor<1x4096x2016xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096xi32>, tensor<1x4096xi32>, tensor<1x4096x576xbf16>
    %1645 = mhlo.copy %1644#18 : tensor<1x4096x576xbf16>
    %1646 = stablehlo.negate %1644#4 : tensor<1x4096x2016xbf16>
    %1647 = stablehlo.exponential %1646 : tensor<1x4096x2016xbf16>
    %1648 = stablehlo.add %1647, %152 : tensor<1x4096x2016xbf16>
    %1649 = stablehlo.divide %152, %1648 : tensor<1x4096x2016xbf16>
    %1650 = stablehlo.multiply %1644#4, %1649 : tensor<1x4096x2016xbf16>
    %1651 = mhlo.copy %1645 : tensor<1x4096x576xbf16>
    %1652 = "stablehlo.all_gather"(%1644#8) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 82, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %1653 = stablehlo.dot_general %1651, %1652, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x2016xbf16>
    %1654 = mhlo.copy %1653 : tensor<1x4096x2016xbf16>
    %1655 = stablehlo.multiply %1650, %1654 : tensor<1x4096x2016xbf16>
    %1656 = "stablehlo.all_gather"(%1644#7) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 83, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %1657 = stablehlo.dot_general %1655, %1656, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %1658 = stablehlo.multiply %1654, %1644#5 : tensor<1x4096x2016xbf16>
    %1659 = stablehlo.multiply %1658, %1649 : tensor<1x4096x2016xbf16>
    %1660 = stablehlo.multiply %1644#4, %1658 : tensor<1x4096x2016xbf16>
    %1661 = stablehlo.subtract %152, %1649 : tensor<1x4096x2016xbf16>
    %1662 = stablehlo.multiply %1649, %1661 : tensor<1x4096x2016xbf16>
    %1663 = stablehlo.multiply %1660, %1662 : tensor<1x4096x2016xbf16>
    %1664 = stablehlo.add %1659, %1663 : tensor<1x4096x2016xbf16>
    %1665 = "stablehlo.all_gather"(%1644#6) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 84, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %1666 = stablehlo.dot_general %1664, %1665, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %1667 = stablehlo.add %1657, %1666 : tensor<1x4096x576xbf16>
    %1668 = mhlo.copy %1667 : tensor<1x4096x576xbf16>
    %1669 = stablehlo.broadcast_in_dim %1644#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %1670 = stablehlo.multiply %1668, %1669 : tensor<1x4096x576xbf16>
    %1671 = stablehlo.convert %1670 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1672 = mhlo.copy %1644#15 : tensor<1x4096x576xbf16>
    %1673 = mhlo.copy %1644#3 : tensor<1x4096x576xbf16>
    %1674 = stablehlo.add %1672, %1673 : tensor<1x4096x576xbf16>
    %1675 = stablehlo.convert %1674 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1676 = stablehlo.multiply %1675, %1675 : tensor<1x4096x576xf32>
    %1677 = stablehlo.reduce(%1676 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1678 = stablehlo.multiply %1677, %34 : tensor<1x4096xf32>
    %1679 = stablehlo.add %1678, %36 : tensor<1x4096xf32>
    %1680 = stablehlo.reshape %1679 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1681 = stablehlo.rsqrt %1680 : tensor<1x4096x1xf32>
    %1682 = stablehlo.reshape %1681 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1683 = stablehlo.broadcast_in_dim %1682, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1684 = stablehlo.multiply %1671, %1683 : tensor<1x4096x576xf32>
    %1685 = stablehlo.multiply %1675, %1671 : tensor<1x4096x576xf32>
    %1686 = stablehlo.reduce(%1685 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1687 = stablehlo.reshape %1686 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1688 = stablehlo.divide %1681, %1680 : tensor<1x4096x1xf32>
    %1689 = stablehlo.multiply %1688, %835 : tensor<1x4096x1xf32>
    %1690 = stablehlo.multiply %1687, %1689 : tensor<1x4096x1xf32>
    %1691 = stablehlo.multiply %1690, %838 : tensor<1x4096x1xf32>
    %1692 = stablehlo.reshape %1691 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1693 = stablehlo.broadcast_in_dim %1692, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1694 = stablehlo.multiply %1675, %1693 : tensor<1x4096x576xf32>
    %1695 = stablehlo.add %1684, %1694 : tensor<1x4096x576xf32>
    %1696 = stablehlo.convert %1695 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %1697 = stablehlo.add %1645, %1696 : tensor<1x4096x576xbf16>
    %1698 = mhlo.copy %1697 : tensor<1x4096x576xbf16>
    %1699 = "stablehlo.all_gather"(%1644#12) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 85, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %1700 = stablehlo.dot_general %1698, %1699, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x9x128xbf16>
    %1701 = mhlo.copy %1700 : tensor<1x4096x9x128xbf16>
    %1702 = stablehlo.broadcast_in_dim %1644#16, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1703 = stablehlo.broadcast_in_dim %1644#16, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1704 = stablehlo.compare  EQ, %1702, %1703 : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %1705 = stablehlo.and %1704, %56 : tensor<1x4096x4096xi1>
    %1706 = stablehlo.select %1705, %58, %59 : tensor<1x4096x4096xi1>, tensor<1x4096x4096xf32>
    %1707 = stablehlo.compare  GE, %1706, %61 : (tensor<1x4096x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xi1>
    %1708 = stablehlo.broadcast_in_dim %1707, dims = [0, 3, 4] : (tensor<1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xi1>
    %1709 = stablehlo.dot_general %1644#1, %1644#0, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1710 = stablehlo.reshape %1709 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %1711 = stablehlo.transpose %1710, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1712 = stablehlo.select %1708, %1711, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %1713 = stablehlo.reduce(%1712 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1714 = stablehlo.broadcast_in_dim %1713, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1715 = stablehlo.subtract %1712, %1714 : tensor<1x9x1x4096x4096xbf16>
    %1716 = stablehlo.exponential %1715 : tensor<1x9x1x4096x4096xbf16>
    %1717 = stablehlo.reshape %1716 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1718 = stablehlo.convert %1717 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %1719 = stablehlo.reduce(%1718 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %1720 = stablehlo.convert %1719 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %1721 = stablehlo.reshape %1720 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %1722 = stablehlo.transpose %1721, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %1723 = stablehlo.reshape %1722 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %1724 = stablehlo.broadcast_in_dim %1723, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %1725 = stablehlo.divide %1701, %1724 : tensor<1x4096x9x128xbf16>
    %1726 = stablehlo.reshape %1725 : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1727 = stablehlo.transpose %1726, dims = [0, 2, 4, 3, 1] {result_layout = dense<[2, 3, 1, 4, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,128,1,4096]{2,3,1,4,0}"} : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %1728 = stablehlo.reshape %1727 {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,128,4096]{2,1,3,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1729 = stablehlo.dot_general %1728, %1717, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1730 = stablehlo.transpose %1729, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{1,3,2,0}"} : (tensor<1x9x128x4096xbf16>) -> tensor<1x4096x9x128xbf16>
    %1731 = mhlo.copy %1730 : tensor<1x4096x9x128xbf16>
    %1732 = "stablehlo.all_gather"(%1644#14) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 86, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1733 = stablehlo.dot_general %1731, %1732, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1734 = stablehlo.broadcast_in_dim %1707, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x9x4096x4096xi1>
    %1735 = stablehlo.dot_general %1728, %1644#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1736 = stablehlo.multiply %1722, %1722 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : tensor<1x4096x9x1xbf16>
    %1737 = stablehlo.divide %937, %1736 : tensor<1x4096x9x1xbf16>
    %1738 = stablehlo.reshape %1737 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %1739 = stablehlo.broadcast_in_dim %1738, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %1740 = stablehlo.multiply %1701, %1739 : tensor<1x4096x9x128xbf16>
    %1741 = stablehlo.dot_general %1644#2, %1717, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1742 = stablehlo.reshape %1741 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %1743 = stablehlo.transpose %1742, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1744 = stablehlo.reshape %1743 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1745 = stablehlo.multiply %1740, %1744 : tensor<1x4096x9x128xbf16>
    %1746 = stablehlo.reduce(%1745 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x4096x9x128xbf16>, tensor<bf16>) -> tensor<1x4096x9xbf16>
    %1747 = stablehlo.negate %1746 : tensor<1x4096x9xbf16>
    %1748 = stablehlo.reshape %1747 : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1xbf16>
    %1749 = stablehlo.transpose %1748, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,1]{3,1,2,0}"} : (tensor<1x4096x9x1xbf16>) -> tensor<1x9x4096x1xbf16>
    %1750 = stablehlo.reshape %1749 : (tensor<1x9x4096x1xbf16>) -> tensor<1x9x4096xbf16>
    %1751 = stablehlo.broadcast_in_dim %1750, dims = [0, 1, 2] : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1752 = stablehlo.add %1735, %1751 : tensor<1x9x4096x4096xbf16>
    %1753 = stablehlo.multiply %1752, %1717 : tensor<1x9x4096x4096xbf16>
    %1754 = stablehlo.compare  EQ, %1712, %1714 : (tensor<1x9x1x4096x4096xbf16>, tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xi1>
    %1755 = stablehlo.reduce(%1753 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1756 = stablehlo.negate %1755 : tensor<1x9x4096xbf16>
    %1757 = stablehlo.convert %1754 : (tensor<1x9x1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xbf16>
    %1758 = stablehlo.reduce(%1757 init: %cst_24) applies stablehlo.add across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1759 = stablehlo.divide %1756, %1758 : tensor<1x9x4096xbf16>
    %1760 = stablehlo.broadcast_in_dim %1759, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1761 = stablehlo.select %1754, %1760, %963 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %1762 = stablehlo.reshape %1761 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1763 = stablehlo.add %1753, %1762 : tensor<1x9x4096x4096xbf16>
    %1764 = stablehlo.select %1734, %1763, %967 : tensor<1x9x4096x4096xi1>, tensor<1x9x4096x4096xbf16>
    %1765 = stablehlo.reshape %1764 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1766 = stablehlo.transpose %1765, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,4096,4096,1]{2,3,4,1,0}"} : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %1767 = stablehlo.reshape %1766 {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,4096]{2,3,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1768 = stablehlo.dot_general %1767, %1644#0, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %1769 = stablehlo.transpose %1768, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{3,1,2,0}"} : (tensor<1x9x4096x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1770 = mhlo.copy %1769 : tensor<1x4096x9x128xbf16>
    %1771 = stablehlo.slice %1770 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1772 = stablehlo.convert %1644#17 : (tensor<1x4096xi32>) -> tensor<1x4096xf32>
    %1773 = stablehlo.broadcast_in_dim %1772, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x1x64xf32>
    %1774 = stablehlo.divide %1773, %77 : tensor<1x4096x1x64xf32>
    %1775 = stablehlo.sine %1774 : tensor<1x4096x1x64xf32>
    %1776 = stablehlo.convert %1775 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %1777 = stablehlo.reshape %1776 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %1778 = stablehlo.broadcast_in_dim %1777, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %1779 = stablehlo.multiply %1771, %1778 : tensor<1x4096x9x64xbf16>
    %1780 = stablehlo.slice %1770 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1781 = stablehlo.cosine %1774 : tensor<1x4096x1x64xf32>
    %1782 = stablehlo.convert %1781 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %1783 = stablehlo.reshape %1782 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %1784 = stablehlo.broadcast_in_dim %1783, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %1785 = stablehlo.multiply %1780, %1784 : tensor<1x4096x9x64xbf16>
    %1786 = stablehlo.add %1779, %1785 : tensor<1x4096x9x64xbf16>
    %1787 = stablehlo.multiply %1771, %1784 : tensor<1x4096x9x64xbf16>
    %1788 = stablehlo.negate %1780 : tensor<1x4096x9x64xbf16>
    %1789 = stablehlo.multiply %1788, %1778 : tensor<1x4096x9x64xbf16>
    %1790 = stablehlo.add %1787, %1789 : tensor<1x4096x9x64xbf16>
    %1791 = stablehlo.concatenate %1786, %1790, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %1792 = "stablehlo.all_gather"(%1644#11) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 87, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1793 = stablehlo.dot_general %1791, %1792, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1794 = stablehlo.add %1733, %1793 : tensor<1x4096x576xbf16>
    %1795 = mhlo.copy %1794 : tensor<1x4096x576xbf16>
    %1796 = stablehlo.dot_general %1767, %1644#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %1797 = stablehlo.reshape %1796 : (tensor<1x9x4096x128xbf16>) -> tensor<1x9x4096x1x128xbf16>
    %1798 = stablehlo.transpose %1797, dims = [0, 2, 1, 3, 4] {result_layout = dense<[4, 3, 1, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{4,3,1,2,0}"} : (tensor<1x9x4096x1x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1799 = stablehlo.reshape %1798 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1800 = mhlo.copy %1799 : tensor<1x4096x9x128xbf16>
    %1801 = stablehlo.slice %1800 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1802 = stablehlo.multiply %1801, %1778 : tensor<1x4096x9x64xbf16>
    %1803 = stablehlo.slice %1800 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1804 = stablehlo.multiply %1803, %1784 : tensor<1x4096x9x64xbf16>
    %1805 = stablehlo.add %1802, %1804 : tensor<1x4096x9x64xbf16>
    %1806 = stablehlo.multiply %1801, %1784 : tensor<1x4096x9x64xbf16>
    %1807 = stablehlo.negate %1803 : tensor<1x4096x9x64xbf16>
    %1808 = stablehlo.multiply %1807, %1778 : tensor<1x4096x9x64xbf16>
    %1809 = stablehlo.add %1806, %1808 : tensor<1x4096x9x64xbf16>
    %1810 = stablehlo.concatenate %1805, %1809, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %1811 = "stablehlo.all_gather"(%1644#13) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 88, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1812 = stablehlo.dot_general %1810, %1811, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1813 = mhlo.copy %1812 : tensor<1x4096x576xbf16>
    %1814 = stablehlo.add %1795, %1813 : tensor<1x4096x576xbf16>
    %1815 = mhlo.copy %1814 : tensor<1x4096x576xbf16>
    %1816 = stablehlo.broadcast_in_dim %1644#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %1817 = stablehlo.multiply %1815, %1816 : tensor<1x4096x576xbf16>
    %1818 = stablehlo.convert %1817 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1819 = stablehlo.convert %1672 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1820 = stablehlo.multiply %1819, %1819 : tensor<1x4096x576xf32>
    %1821 = stablehlo.reduce(%1820 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1822 = stablehlo.multiply %1821, %34 : tensor<1x4096xf32>
    %1823 = stablehlo.add %1822, %36 : tensor<1x4096xf32>
    %1824 = stablehlo.reshape %1823 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1825 = stablehlo.rsqrt %1824 : tensor<1x4096x1xf32>
    %1826 = stablehlo.reshape %1825 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1827 = stablehlo.broadcast_in_dim %1826, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1828 = stablehlo.multiply %1818, %1827 : tensor<1x4096x576xf32>
    %1829 = stablehlo.multiply %1819, %1818 : tensor<1x4096x576xf32>
    %1830 = stablehlo.reduce(%1829 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1831 = stablehlo.reshape %1830 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1832 = stablehlo.divide %1825, %1824 : tensor<1x4096x1xf32>
    %1833 = stablehlo.multiply %1832, %835 : tensor<1x4096x1xf32>
    %1834 = stablehlo.multiply %1831, %1833 : tensor<1x4096x1xf32>
    %1835 = stablehlo.multiply %1834, %838 : tensor<1x4096x1xf32>
    %1836 = stablehlo.reshape %1835 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1837 = stablehlo.broadcast_in_dim %1836, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1838 = stablehlo.multiply %1819, %1837 : tensor<1x4096x576xf32>
    %1839 = stablehlo.add %1828, %1838 : tensor<1x4096x576xf32>
    %1840 = stablehlo.convert %1839 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %1841 = stablehlo.add %1697, %1840 : tensor<1x4096x576xbf16>
    %1842 = mhlo.copy %1841 : tensor<1x4096x576xbf16>
    %1843:19 = stablehlo.optimization_barrier %211, %198, %185, %236, %252, %260, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %167, %arg203, %arg202, %1842 : tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>, tensor<1x4096x2016xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096xi32>, tensor<1x4096xi32>, tensor<1x4096x576xbf16>
    %1844 = mhlo.copy %1843#18 : tensor<1x4096x576xbf16>
    %1845 = stablehlo.negate %1843#4 : tensor<1x4096x2016xbf16>
    %1846 = stablehlo.exponential %1845 : tensor<1x4096x2016xbf16>
    %1847 = stablehlo.add %1846, %152 : tensor<1x4096x2016xbf16>
    %1848 = stablehlo.divide %152, %1847 : tensor<1x4096x2016xbf16>
    %1849 = stablehlo.multiply %1843#4, %1848 : tensor<1x4096x2016xbf16>
    %1850 = mhlo.copy %1844 : tensor<1x4096x576xbf16>
    %1851 = "stablehlo.all_gather"(%1843#8) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 89, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %1852 = stablehlo.dot_general %1850, %1851, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x2016xbf16>
    %1853 = mhlo.copy %1852 : tensor<1x4096x2016xbf16>
    %1854 = stablehlo.multiply %1849, %1853 : tensor<1x4096x2016xbf16>
    %1855 = "stablehlo.all_gather"(%1843#7) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 90, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %1856 = stablehlo.dot_general %1854, %1855, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %1857 = stablehlo.multiply %1853, %1843#5 : tensor<1x4096x2016xbf16>
    %1858 = stablehlo.multiply %1857, %1848 : tensor<1x4096x2016xbf16>
    %1859 = stablehlo.multiply %1843#4, %1857 : tensor<1x4096x2016xbf16>
    %1860 = stablehlo.subtract %152, %1848 : tensor<1x4096x2016xbf16>
    %1861 = stablehlo.multiply %1848, %1860 : tensor<1x4096x2016xbf16>
    %1862 = stablehlo.multiply %1859, %1861 : tensor<1x4096x2016xbf16>
    %1863 = stablehlo.add %1858, %1862 : tensor<1x4096x2016xbf16>
    %1864 = "stablehlo.all_gather"(%1843#6) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 91, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %1865 = stablehlo.dot_general %1863, %1864, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %1866 = stablehlo.add %1856, %1865 : tensor<1x4096x576xbf16>
    %1867 = mhlo.copy %1866 : tensor<1x4096x576xbf16>
    %1868 = stablehlo.broadcast_in_dim %1843#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %1869 = stablehlo.multiply %1867, %1868 : tensor<1x4096x576xbf16>
    %1870 = stablehlo.convert %1869 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1871 = mhlo.copy %1843#15 : tensor<1x4096x576xbf16>
    %1872 = mhlo.copy %1843#3 : tensor<1x4096x576xbf16>
    %1873 = stablehlo.add %1871, %1872 : tensor<1x4096x576xbf16>
    %1874 = stablehlo.convert %1873 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %1875 = stablehlo.multiply %1874, %1874 : tensor<1x4096x576xf32>
    %1876 = stablehlo.reduce(%1875 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1877 = stablehlo.multiply %1876, %34 : tensor<1x4096xf32>
    %1878 = stablehlo.add %1877, %36 : tensor<1x4096xf32>
    %1879 = stablehlo.reshape %1878 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1880 = stablehlo.rsqrt %1879 : tensor<1x4096x1xf32>
    %1881 = stablehlo.reshape %1880 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1882 = stablehlo.broadcast_in_dim %1881, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1883 = stablehlo.multiply %1870, %1882 : tensor<1x4096x576xf32>
    %1884 = stablehlo.multiply %1874, %1870 : tensor<1x4096x576xf32>
    %1885 = stablehlo.reduce(%1884 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %1886 = stablehlo.reshape %1885 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %1887 = stablehlo.divide %1880, %1879 : tensor<1x4096x1xf32>
    %1888 = stablehlo.multiply %1887, %835 : tensor<1x4096x1xf32>
    %1889 = stablehlo.multiply %1886, %1888 : tensor<1x4096x1xf32>
    %1890 = stablehlo.multiply %1889, %838 : tensor<1x4096x1xf32>
    %1891 = stablehlo.reshape %1890 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %1892 = stablehlo.broadcast_in_dim %1891, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %1893 = stablehlo.multiply %1874, %1892 : tensor<1x4096x576xf32>
    %1894 = stablehlo.add %1883, %1893 : tensor<1x4096x576xf32>
    %1895 = stablehlo.convert %1894 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %1896 = stablehlo.add %1844, %1895 : tensor<1x4096x576xbf16>
    %1897 = mhlo.copy %1896 : tensor<1x4096x576xbf16>
    %1898 = "stablehlo.all_gather"(%1843#12) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 92, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %1899 = stablehlo.dot_general %1897, %1898, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x9x128xbf16>
    %1900 = mhlo.copy %1899 : tensor<1x4096x9x128xbf16>
    %1901 = stablehlo.broadcast_in_dim %1843#16, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1902 = stablehlo.broadcast_in_dim %1843#16, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1903 = stablehlo.compare  EQ, %1901, %1902 : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %1904 = stablehlo.and %1903, %56 : tensor<1x4096x4096xi1>
    %1905 = stablehlo.select %1904, %58, %59 : tensor<1x4096x4096xi1>, tensor<1x4096x4096xf32>
    %1906 = stablehlo.compare  GE, %1905, %61 : (tensor<1x4096x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xi1>
    %1907 = stablehlo.broadcast_in_dim %1906, dims = [0, 3, 4] : (tensor<1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xi1>
    %1908 = stablehlo.dot_general %1843#1, %1843#0, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1909 = stablehlo.reshape %1908 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %1910 = stablehlo.transpose %1909, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1911 = stablehlo.select %1907, %1910, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %1912 = stablehlo.reduce(%1911 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1913 = stablehlo.broadcast_in_dim %1912, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1914 = stablehlo.subtract %1911, %1913 : tensor<1x9x1x4096x4096xbf16>
    %1915 = stablehlo.exponential %1914 : tensor<1x9x1x4096x4096xbf16>
    %1916 = stablehlo.reshape %1915 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1917 = stablehlo.convert %1916 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %1918 = stablehlo.reduce(%1917 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %1919 = stablehlo.convert %1918 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %1920 = stablehlo.reshape %1919 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %1921 = stablehlo.transpose %1920, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %1922 = stablehlo.reshape %1921 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %1923 = stablehlo.broadcast_in_dim %1922, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %1924 = stablehlo.divide %1900, %1923 : tensor<1x4096x9x128xbf16>
    %1925 = stablehlo.reshape %1924 : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1926 = stablehlo.transpose %1925, dims = [0, 2, 4, 3, 1] {result_layout = dense<[2, 3, 1, 4, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,128,1,4096]{2,3,1,4,0}"} : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %1927 = stablehlo.reshape %1926 {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,128,4096]{2,1,3,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1928 = stablehlo.dot_general %1927, %1916, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1929 = stablehlo.transpose %1928, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{1,3,2,0}"} : (tensor<1x9x128x4096xbf16>) -> tensor<1x4096x9x128xbf16>
    %1930 = mhlo.copy %1929 : tensor<1x4096x9x128xbf16>
    %1931 = "stablehlo.all_gather"(%1843#14) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 93, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1932 = stablehlo.dot_general %1930, %1931, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1933 = stablehlo.broadcast_in_dim %1906, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x9x4096x4096xi1>
    %1934 = stablehlo.dot_general %1927, %1843#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1935 = stablehlo.multiply %1921, %1921 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : tensor<1x4096x9x1xbf16>
    %1936 = stablehlo.divide %937, %1935 : tensor<1x4096x9x1xbf16>
    %1937 = stablehlo.reshape %1936 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %1938 = stablehlo.broadcast_in_dim %1937, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %1939 = stablehlo.multiply %1900, %1938 : tensor<1x4096x9x128xbf16>
    %1940 = stablehlo.dot_general %1843#2, %1916, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %1941 = stablehlo.reshape %1940 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %1942 = stablehlo.transpose %1941, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1943 = stablehlo.reshape %1942 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1944 = stablehlo.multiply %1939, %1943 : tensor<1x4096x9x128xbf16>
    %1945 = stablehlo.reduce(%1944 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x4096x9x128xbf16>, tensor<bf16>) -> tensor<1x4096x9xbf16>
    %1946 = stablehlo.negate %1945 : tensor<1x4096x9xbf16>
    %1947 = stablehlo.reshape %1946 : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1xbf16>
    %1948 = stablehlo.transpose %1947, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,1]{3,1,2,0}"} : (tensor<1x4096x9x1xbf16>) -> tensor<1x9x4096x1xbf16>
    %1949 = stablehlo.reshape %1948 : (tensor<1x9x4096x1xbf16>) -> tensor<1x9x4096xbf16>
    %1950 = stablehlo.broadcast_in_dim %1949, dims = [0, 1, 2] : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1951 = stablehlo.add %1934, %1950 : tensor<1x9x4096x4096xbf16>
    %1952 = stablehlo.multiply %1951, %1916 : tensor<1x9x4096x4096xbf16>
    %1953 = stablehlo.compare  EQ, %1911, %1913 : (tensor<1x9x1x4096x4096xbf16>, tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xi1>
    %1954 = stablehlo.reduce(%1952 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1955 = stablehlo.negate %1954 : tensor<1x9x4096xbf16>
    %1956 = stablehlo.convert %1953 : (tensor<1x9x1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xbf16>
    %1957 = stablehlo.reduce(%1956 init: %cst_24) applies stablehlo.add across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %1958 = stablehlo.divide %1955, %1957 : tensor<1x9x4096xbf16>
    %1959 = stablehlo.broadcast_in_dim %1958, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1960 = stablehlo.select %1953, %1959, %963 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %1961 = stablehlo.reshape %1960 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1962 = stablehlo.add %1952, %1961 : tensor<1x9x4096x4096xbf16>
    %1963 = stablehlo.select %1933, %1962, %967 : tensor<1x9x4096x4096xi1>, tensor<1x9x4096x4096xbf16>
    %1964 = stablehlo.reshape %1963 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %1965 = stablehlo.transpose %1964, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,4096,4096,1]{2,3,4,1,0}"} : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %1966 = stablehlo.reshape %1965 {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,4096]{2,3,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x4096x4096xbf16>
    %1967 = stablehlo.dot_general %1966, %1843#0, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %1968 = stablehlo.transpose %1967, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{3,1,2,0}"} : (tensor<1x9x4096x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1969 = mhlo.copy %1968 : tensor<1x4096x9x128xbf16>
    %1970 = stablehlo.slice %1969 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1971 = stablehlo.convert %1843#17 : (tensor<1x4096xi32>) -> tensor<1x4096xf32>
    %1972 = stablehlo.broadcast_in_dim %1971, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x1x64xf32>
    %1973 = stablehlo.divide %1972, %77 : tensor<1x4096x1x64xf32>
    %1974 = stablehlo.sine %1973 : tensor<1x4096x1x64xf32>
    %1975 = stablehlo.convert %1974 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %1976 = stablehlo.reshape %1975 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %1977 = stablehlo.broadcast_in_dim %1976, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %1978 = stablehlo.multiply %1970, %1977 : tensor<1x4096x9x64xbf16>
    %1979 = stablehlo.slice %1969 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %1980 = stablehlo.cosine %1973 : tensor<1x4096x1x64xf32>
    %1981 = stablehlo.convert %1980 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %1982 = stablehlo.reshape %1981 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %1983 = stablehlo.broadcast_in_dim %1982, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %1984 = stablehlo.multiply %1979, %1983 : tensor<1x4096x9x64xbf16>
    %1985 = stablehlo.add %1978, %1984 : tensor<1x4096x9x64xbf16>
    %1986 = stablehlo.multiply %1970, %1983 : tensor<1x4096x9x64xbf16>
    %1987 = stablehlo.negate %1979 : tensor<1x4096x9x64xbf16>
    %1988 = stablehlo.multiply %1987, %1977 : tensor<1x4096x9x64xbf16>
    %1989 = stablehlo.add %1986, %1988 : tensor<1x4096x9x64xbf16>
    %1990 = stablehlo.concatenate %1985, %1989, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %1991 = "stablehlo.all_gather"(%1843#11) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 94, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %1992 = stablehlo.dot_general %1990, %1991, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %1993 = stablehlo.add %1932, %1992 : tensor<1x4096x576xbf16>
    %1994 = mhlo.copy %1993 : tensor<1x4096x576xbf16>
    %1995 = stablehlo.dot_general %1966, %1843#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %1996 = stablehlo.reshape %1995 : (tensor<1x9x4096x128xbf16>) -> tensor<1x9x4096x1x128xbf16>
    %1997 = stablehlo.transpose %1996, dims = [0, 2, 1, 3, 4] {result_layout = dense<[4, 3, 1, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{4,3,1,2,0}"} : (tensor<1x9x4096x1x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %1998 = stablehlo.reshape %1997 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %1999 = mhlo.copy %1998 : tensor<1x4096x9x128xbf16>
    %2000 = stablehlo.slice %1999 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %2001 = stablehlo.multiply %2000, %1977 : tensor<1x4096x9x64xbf16>
    %2002 = stablehlo.slice %1999 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %2003 = stablehlo.multiply %2002, %1983 : tensor<1x4096x9x64xbf16>
    %2004 = stablehlo.add %2001, %2003 : tensor<1x4096x9x64xbf16>
    %2005 = stablehlo.multiply %2000, %1983 : tensor<1x4096x9x64xbf16>
    %2006 = stablehlo.negate %2002 : tensor<1x4096x9x64xbf16>
    %2007 = stablehlo.multiply %2006, %1977 : tensor<1x4096x9x64xbf16>
    %2008 = stablehlo.add %2005, %2007 : tensor<1x4096x9x64xbf16>
    %2009 = stablehlo.concatenate %2004, %2008, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %2010 = "stablehlo.all_gather"(%1843#13) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 95, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2011 = stablehlo.dot_general %2009, %2010, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %2012 = mhlo.copy %2011 : tensor<1x4096x576xbf16>
    %2013 = stablehlo.add %1994, %2012 : tensor<1x4096x576xbf16>
    %2014 = mhlo.copy %2013 : tensor<1x4096x576xbf16>
    %2015 = stablehlo.broadcast_in_dim %1843#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %2016 = stablehlo.multiply %2014, %2015 : tensor<1x4096x576xbf16>
    %2017 = stablehlo.convert %2016 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %2018 = stablehlo.convert %1871 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %2019 = stablehlo.multiply %2018, %2018 : tensor<1x4096x576xf32>
    %2020 = stablehlo.reduce(%2019 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %2021 = stablehlo.multiply %2020, %34 : tensor<1x4096xf32>
    %2022 = stablehlo.add %2021, %36 : tensor<1x4096xf32>
    %2023 = stablehlo.reshape %2022 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %2024 = stablehlo.rsqrt %2023 : tensor<1x4096x1xf32>
    %2025 = stablehlo.reshape %2024 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %2026 = stablehlo.broadcast_in_dim %2025, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %2027 = stablehlo.multiply %2017, %2026 : tensor<1x4096x576xf32>
    %2028 = stablehlo.multiply %2018, %2017 : tensor<1x4096x576xf32>
    %2029 = stablehlo.reduce(%2028 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %2030 = stablehlo.reshape %2029 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %2031 = stablehlo.divide %2024, %2023 : tensor<1x4096x1xf32>
    %2032 = stablehlo.multiply %2031, %835 : tensor<1x4096x1xf32>
    %2033 = stablehlo.multiply %2030, %2032 : tensor<1x4096x1xf32>
    %2034 = stablehlo.multiply %2033, %838 : tensor<1x4096x1xf32>
    %2035 = stablehlo.reshape %2034 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %2036 = stablehlo.broadcast_in_dim %2035, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %2037 = stablehlo.multiply %2018, %2036 : tensor<1x4096x576xf32>
    %2038 = stablehlo.add %2027, %2037 : tensor<1x4096x576xf32>
    %2039 = stablehlo.convert %2038 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2040 = stablehlo.add %1896, %2039 : tensor<1x4096x576xbf16>
    %2041 = mhlo.copy %2040 : tensor<1x4096x576xbf16>
    %2042:19 = stablehlo.optimization_barrier %109, %96, %49, %135, %151, %160, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %29, %arg203, %arg202, %2041 : tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>, tensor<1x4096x2016xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<1x4096x576xbf16>, tensor<1x4096xi32>, tensor<1x4096xi32>, tensor<1x4096x576xbf16>
    %2043 = mhlo.copy %2042#15 : tensor<1x4096x576xbf16>
    %2044 = mhlo.copy %2042#3 : tensor<1x4096x576xbf16>
    %2045 = stablehlo.add %2043, %2044 : tensor<1x4096x576xbf16>
    %2046 = stablehlo.convert %2045 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %2047 = stablehlo.multiply %2046, %2046 : tensor<1x4096x576xf32>
    %2048 = stablehlo.reduce(%2047 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %2049 = stablehlo.multiply %2048, %34 : tensor<1x4096xf32>
    %2050 = stablehlo.add %2049, %36 : tensor<1x4096xf32>
    %2051 = stablehlo.reshape %2050 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %2052 = stablehlo.rsqrt %2051 : tensor<1x4096x1xf32>
    %2053 = stablehlo.reshape %2052 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %2054 = stablehlo.broadcast_in_dim %2053, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %2055 = stablehlo.multiply %2046, %2054 : tensor<1x4096x576xf32>
    %2056 = stablehlo.convert %2055 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2057 = stablehlo.broadcast_in_dim %2042#9, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %2058 = stablehlo.multiply %2056, %2057 : tensor<1x4096x576xbf16>
    %2059 = mhlo.copy %2058 : tensor<1x4096x576xbf16>
    %2060 = mhlo.copy %2042#18 : tensor<1x4096x576xbf16>
    %2061 = mhlo.copy %2060 : tensor<1x4096x576xbf16>
    %2062 = "stablehlo.all_gather"(%2042#8) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 96, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<2016x144xbf16>) -> tensor<2016x576xbf16>
    %2063 = stablehlo.dot_general %2061, %2062, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<2016x576xbf16>) -> tensor<1x4096x2016xbf16>
    %2064 = mhlo.copy %2063 : tensor<1x4096x2016xbf16>
    %2065 = stablehlo.multiply %2064, %2042#5 : tensor<1x4096x2016xbf16>
    %2066 = stablehlo.negate %2042#4 : tensor<1x4096x2016xbf16>
    %2067 = stablehlo.exponential %2066 : tensor<1x4096x2016xbf16>
    %2068 = stablehlo.add %2067, %152 : tensor<1x4096x2016xbf16>
    %2069 = stablehlo.divide %152, %2068 : tensor<1x4096x2016xbf16>
    %2070 = stablehlo.multiply %2065, %2069 : tensor<1x4096x2016xbf16>
    %2071 = stablehlo.multiply %2042#4, %2065 : tensor<1x4096x2016xbf16>
    %2072 = stablehlo.subtract %152, %2069 : tensor<1x4096x2016xbf16>
    %2073 = stablehlo.multiply %2069, %2072 : tensor<1x4096x2016xbf16>
    %2074 = stablehlo.multiply %2071, %2073 : tensor<1x4096x2016xbf16>
    %2075 = stablehlo.add %2070, %2074 : tensor<1x4096x2016xbf16>
    %2076 = stablehlo.dot_general %2059, %2075, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2077 = "stablehlo.all_reduce"(%2076) <{channel_handle = #stablehlo.channel_handle<handle = 97, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %c_27 = stablehlo.constant dense<[0, 144, 288, 432]> : tensor<4xi32>
    %2078 = stablehlo.partition_id : tensor<ui32>
    %2079 = stablehlo.dynamic_slice %c_27, %2078, sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
    %2080 = stablehlo.reshape %2079 : (tensor<1xi32>) -> tensor<i32>
    %2081 = stablehlo.dynamic_slice %2077, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2082 = stablehlo.multiply %2081, %2081 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2083 = stablehlo.convert %2082 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2084 = stablehlo.reduce(%2083 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2085 = "stablehlo.all_reduce"(%2084) <{channel_handle = #stablehlo.channel_handle<handle = 98, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2086 = stablehlo.convert %2085 : (tensor<f32>) -> tensor<bf16>
    %2087 = stablehlo.add %827, %2086 : tensor<bf16>
    %2088 = stablehlo.multiply %2042#4, %2069 : tensor<1x4096x2016xbf16>
    %2089 = stablehlo.multiply %2088, %2064 : tensor<1x4096x2016xbf16>
    %2090 = stablehlo.dot_general %2059, %2089, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2091 = "stablehlo.all_reduce"(%2090) <{channel_handle = #stablehlo.channel_handle<handle = 99, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %2092 = stablehlo.dynamic_slice %2091, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2093 = stablehlo.multiply %2092, %2092 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2094 = stablehlo.convert %2093 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2095 = stablehlo.reduce(%2094 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2096 = "stablehlo.all_reduce"(%2095) <{channel_handle = #stablehlo.channel_handle<handle = 100, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2097 = stablehlo.convert %2096 : (tensor<f32>) -> tensor<bf16>
    %2098 = stablehlo.add %2087, %2097 : tensor<bf16>
    %2099 = stablehlo.multiply %2088, %2042#5 : tensor<1x4096x2016xbf16>
    %2100 = mhlo.copy %2099 : tensor<1x4096x2016xbf16>
    %2101 = stablehlo.dot_general %2100, %2061, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<1x4096x576xbf16>) -> tensor<2016x576xbf16>
    %2102 = "stablehlo.all_reduce"(%2101) <{channel_handle = #stablehlo.channel_handle<handle = 101, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<2016x576xbf16>) -> tensor<2016x576xbf16>
    %2103 = stablehlo.dynamic_slice %2102, %c_1, %2080, sizes = [2016, 144] : (tensor<2016x576xbf16>, tensor<i32>, tensor<i32>) -> tensor<2016x144xbf16>
    %2104 = stablehlo.multiply %2103, %2103 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %2105 = stablehlo.convert %2104 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %2106 = stablehlo.reduce(%2105 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %2107 = "stablehlo.all_reduce"(%2106) <{channel_handle = #stablehlo.channel_handle<handle = 102, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2108 = stablehlo.convert %2107 : (tensor<f32>) -> tensor<bf16>
    %2109 = stablehlo.add %2098, %2108 : tensor<bf16>
    %2110 = "stablehlo.all_gather"(%2042#7) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 103, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %2111 = stablehlo.dot_general %2089, %2110, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %2112 = "stablehlo.all_gather"(%2042#6) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 104, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x2016xbf16>) -> tensor<576x2016xbf16>
    %2113 = stablehlo.dot_general %2075, %2112, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<576x2016xbf16>) -> tensor<1x4096x576xbf16>
    %2114 = stablehlo.add %2111, %2113 : tensor<1x4096x576xbf16>
    %2115 = mhlo.copy %2114 : tensor<1x4096x576xbf16>
    %2116 = stablehlo.multiply %2056, %2115 : tensor<1x4096x576xbf16>
    %2117 = stablehlo.reduce(%2116 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2118 = "stablehlo.all_reduce"(%2117) <{channel_handle = #stablehlo.channel_handle<handle = 105, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2119 = stablehlo.multiply %2118, %2118 : tensor<576xbf16>
    %2120 = stablehlo.convert %2119 : (tensor<576xbf16>) -> tensor<576xf32>
    %2121 = stablehlo.reduce(%2120 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2122 = stablehlo.convert %2121 : (tensor<f32>) -> tensor<bf16>
    %2123 = stablehlo.add %2109, %2122 : tensor<bf16>
    %2124 = stablehlo.convert %2043 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %2125 = stablehlo.multiply %2124, %2124 : tensor<1x4096x576xf32>
    %2126 = stablehlo.reduce(%2125 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %2127 = stablehlo.multiply %2126, %34 : tensor<1x4096xf32>
    %2128 = stablehlo.add %2127, %36 : tensor<1x4096xf32>
    %2129 = stablehlo.reshape %2128 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %2130 = stablehlo.rsqrt %2129 : tensor<1x4096x1xf32>
    %2131 = stablehlo.reshape %2130 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %2132 = stablehlo.broadcast_in_dim %2131, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %2133 = stablehlo.multiply %2124, %2132 : tensor<1x4096x576xf32>
    %2134 = stablehlo.convert %2133 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2135 = stablehlo.multiply %2115, %2057 : tensor<1x4096x576xbf16>
    %2136 = stablehlo.convert %2135 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %2137 = stablehlo.multiply %2136, %2054 : tensor<1x4096x576xf32>
    %2138 = stablehlo.multiply %2046, %2136 : tensor<1x4096x576xf32>
    %2139 = stablehlo.reduce(%2138 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %2140 = stablehlo.reshape %2139 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %2141 = stablehlo.divide %2052, %2051 : tensor<1x4096x1xf32>
    %2142 = stablehlo.multiply %2141, %835 : tensor<1x4096x1xf32>
    %2143 = stablehlo.multiply %2140, %2142 : tensor<1x4096x1xf32>
    %2144 = stablehlo.multiply %2143, %838 : tensor<1x4096x1xf32>
    %2145 = stablehlo.reshape %2144 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %2146 = stablehlo.broadcast_in_dim %2145, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %2147 = stablehlo.multiply %2046, %2146 : tensor<1x4096x576xf32>
    %2148 = stablehlo.add %2137, %2147 : tensor<1x4096x576xf32>
    %2149 = stablehlo.convert %2148 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2150 = stablehlo.add %2060, %2149 : tensor<1x4096x576xbf16>
    %2151 = mhlo.copy %2150 : tensor<1x4096x576xbf16>
    %2152 = "stablehlo.all_gather"(%2042#12) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 106, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<9x128x144xbf16>) -> tensor<9x128x576xbf16>
    %2153 = stablehlo.dot_general %2151, %2152, contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<9x128x576xbf16>) -> tensor<1x4096x9x128xbf16>
    %2154 = mhlo.copy %2153 : tensor<1x4096x9x128xbf16>
    %2155 = stablehlo.broadcast_in_dim %2042#16, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %2156 = stablehlo.broadcast_in_dim %2042#16, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x4096x4096xi32>
    %2157 = stablehlo.compare  EQ, %2155, %2156 : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %2158 = stablehlo.and %2157, %56 : tensor<1x4096x4096xi1>
    %2159 = stablehlo.select %2158, %58, %59 : tensor<1x4096x4096xi1>, tensor<1x4096x4096xf32>
    %2160 = stablehlo.compare  GE, %2159, %61 : (tensor<1x4096x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xi1>
    %2161 = stablehlo.broadcast_in_dim %2160, dims = [0, 3, 4] : (tensor<1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xi1>
    %2162 = stablehlo.dot_general %2042#1, %2042#0, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %2163 = stablehlo.reshape %2162 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %2164 = stablehlo.transpose %2163, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,1,4096,4096]{2,3,4,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %2165 = stablehlo.select %2161, %2164, %113 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %2166 = stablehlo.reduce(%2165 init: %cst_20) applies stablehlo.maximum across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %2167 = stablehlo.broadcast_in_dim %2166, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %2168 = stablehlo.subtract %2165, %2167 : tensor<1x9x1x4096x4096xbf16>
    %2169 = stablehlo.exponential %2168 : tensor<1x9x1x4096x4096xbf16>
    %2170 = stablehlo.reshape %2169 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %2171 = stablehlo.convert %2170 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x4096x4096xf32>
    %2172 = stablehlo.reduce(%2171 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xf32>, tensor<f32>) -> tensor<1x9x4096xf32>
    %2173 = stablehlo.convert %2172 : (tensor<1x9x4096xf32>) -> tensor<1x9x4096xbf16>
    %2174 = stablehlo.reshape %2173 : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x1xbf16>
    %2175 = stablehlo.transpose %2174, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : (tensor<1x9x4096x1xbf16>) -> tensor<1x4096x9x1xbf16>
    %2176 = stablehlo.reshape %2175 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %2177 = stablehlo.broadcast_in_dim %2176, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %2178 = stablehlo.divide %2154, %2177 : tensor<1x4096x9x128xbf16>
    %2179 = stablehlo.reshape %2178 : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %2180 = stablehlo.transpose %2179, dims = [0, 2, 4, 3, 1] {result_layout = dense<[2, 3, 1, 4, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,128,1,4096]{2,3,1,4,0}"} : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %2181 = stablehlo.reshape %2180 {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,128,4096]{2,1,3,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %2182 = stablehlo.dot_general %2181, %2170, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %2183 = stablehlo.transpose %2182, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{1,3,2,0}"} : (tensor<1x9x128x4096xbf16>) -> tensor<1x4096x9x128xbf16>
    %2184 = mhlo.copy %2183 : tensor<1x4096x9x128xbf16>
    %2185 = "stablehlo.all_gather"(%2042#14) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 107, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2186 = stablehlo.dot_general %2184, %2185, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %2187 = stablehlo.broadcast_in_dim %2160, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x9x4096x4096xi1>
    %2188 = stablehlo.dot_general %2181, %2042#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x9x128x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x4096xbf16>
    %2189 = stablehlo.multiply %2175, %2175 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,1]{3,1,2,0}"} : tensor<1x4096x9x1xbf16>
    %2190 = stablehlo.divide %937, %2189 : tensor<1x4096x9x1xbf16>
    %2191 = stablehlo.reshape %2190 : (tensor<1x4096x9x1xbf16>) -> tensor<1x4096x9xbf16>
    %2192 = stablehlo.broadcast_in_dim %2191, dims = [0, 1, 2] : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x128xbf16>
    %2193 = stablehlo.multiply %2154, %2192 : tensor<1x4096x9x128xbf16>
    %2194 = stablehlo.dot_general %2042#2, %2170, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x9x4096x4096xbf16>) -> tensor<1x9x128x4096xbf16>
    %2195 = stablehlo.reshape %2194 : (tensor<1x9x128x4096xbf16>) -> tensor<1x9x128x1x4096xbf16>
    %2196 = stablehlo.transpose %2195, dims = [0, 4, 1, 3, 2] {result_layout = dense<[1, 3, 4, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{1,3,4,2,0}"} : (tensor<1x9x128x1x4096xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %2197 = stablehlo.reshape %2196 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %2198 = stablehlo.multiply %2193, %2197 : tensor<1x4096x9x128xbf16>
    %2199 = stablehlo.reduce(%2198 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x4096x9x128xbf16>, tensor<bf16>) -> tensor<1x4096x9xbf16>
    %2200 = stablehlo.negate %2199 : tensor<1x4096x9xbf16>
    %2201 = stablehlo.reshape %2200 : (tensor<1x4096x9xbf16>) -> tensor<1x4096x9x1xbf16>
    %2202 = stablehlo.transpose %2201, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,1]{3,1,2,0}"} : (tensor<1x4096x9x1xbf16>) -> tensor<1x9x4096x1xbf16>
    %2203 = stablehlo.reshape %2202 : (tensor<1x9x4096x1xbf16>) -> tensor<1x9x4096xbf16>
    %2204 = stablehlo.broadcast_in_dim %2203, dims = [0, 1, 2] : (tensor<1x9x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %2205 = stablehlo.add %2188, %2204 : tensor<1x9x4096x4096xbf16>
    %2206 = stablehlo.multiply %2205, %2170 : tensor<1x9x4096x4096xbf16>
    %2207 = stablehlo.compare  EQ, %2165, %2167 : (tensor<1x9x1x4096x4096xbf16>, tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xi1>
    %2208 = stablehlo.reduce(%2206 init: %cst_24) applies stablehlo.add across dimensions = [3] : (tensor<1x9x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %2209 = stablehlo.negate %2208 : tensor<1x9x4096xbf16>
    %2210 = stablehlo.convert %2207 : (tensor<1x9x1x4096x4096xi1>) -> tensor<1x9x1x4096x4096xbf16>
    %2211 = stablehlo.reduce(%2210 init: %cst_24) applies stablehlo.add across dimensions = [2, 4] : (tensor<1x9x1x4096x4096xbf16>, tensor<bf16>) -> tensor<1x9x4096xbf16>
    %2212 = stablehlo.divide %2209, %2211 : tensor<1x9x4096xbf16>
    %2213 = stablehlo.broadcast_in_dim %2212, dims = [0, 1, 3] : (tensor<1x9x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %2214 = stablehlo.select %2207, %2213, %963 : tensor<1x9x1x4096x4096xi1>, tensor<1x9x1x4096x4096xbf16>
    %2215 = stablehlo.reshape %2214 : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096xbf16>
    %2216 = stablehlo.add %2206, %2215 : tensor<1x9x4096x4096xbf16>
    %2217 = stablehlo.select %2187, %2216, %967 : tensor<1x9x4096x4096xi1>, tensor<1x9x4096x4096xbf16>
    %2218 = stablehlo.reshape %2217 : (tensor<1x9x4096x4096xbf16>) -> tensor<1x9x1x4096x4096xbf16>
    %2219 = stablehlo.transpose %2218, dims = [0, 1, 4, 3, 2] {result_layout = dense<[2, 3, 4, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,9,4096,4096,1]{2,3,4,1,0}"} : (tensor<1x9x1x4096x4096xbf16>) -> tensor<1x9x4096x4096x1xbf16>
    %2220 = stablehlo.reshape %2219 {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "bf16[1,9,4096,4096]{2,3,1,0}"} : (tensor<1x9x4096x4096x1xbf16>) -> tensor<1x9x4096x4096xbf16>
    %2221 = stablehlo.dot_general %2220, %2042#0, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %2222 = stablehlo.transpose %2221, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,4096,9,128]{3,1,2,0}"} : (tensor<1x9x4096x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %2223 = mhlo.copy %2222 : tensor<1x4096x9x128xbf16>
    %2224 = stablehlo.slice %2223 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %2225 = stablehlo.convert %2042#17 : (tensor<1x4096xi32>) -> tensor<1x4096xf32>
    %2226 = stablehlo.broadcast_in_dim %2225, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x1x64xf32>
    %2227 = stablehlo.divide %2226, %77 : tensor<1x4096x1x64xf32>
    %2228 = stablehlo.sine %2227 : tensor<1x4096x1x64xf32>
    %2229 = stablehlo.convert %2228 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %2230 = stablehlo.reshape %2229 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %2231 = stablehlo.broadcast_in_dim %2230, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %2232 = stablehlo.multiply %2224, %2231 : tensor<1x4096x9x64xbf16>
    %2233 = stablehlo.slice %2223 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %2234 = stablehlo.cosine %2227 : tensor<1x4096x1x64xf32>
    %2235 = stablehlo.convert %2234 : (tensor<1x4096x1x64xf32>) -> tensor<1x4096x1x64xbf16>
    %2236 = stablehlo.reshape %2235 : (tensor<1x4096x1x64xbf16>) -> tensor<1x4096x64xbf16>
    %2237 = stablehlo.broadcast_in_dim %2236, dims = [0, 1, 3] : (tensor<1x4096x64xbf16>) -> tensor<1x4096x9x64xbf16>
    %2238 = stablehlo.multiply %2233, %2237 : tensor<1x4096x9x64xbf16>
    %2239 = stablehlo.add %2232, %2238 : tensor<1x4096x9x64xbf16>
    %2240 = stablehlo.multiply %2224, %2237 : tensor<1x4096x9x64xbf16>
    %2241 = stablehlo.negate %2233 : tensor<1x4096x9x64xbf16>
    %2242 = stablehlo.multiply %2241, %2231 : tensor<1x4096x9x64xbf16>
    %2243 = stablehlo.add %2240, %2242 : tensor<1x4096x9x64xbf16>
    %2244 = stablehlo.concatenate %2239, %2243, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %2245 = "stablehlo.all_gather"(%2042#11) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 108, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2246 = stablehlo.dot_general %2244, %2245, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %2247 = stablehlo.add %2186, %2246 : tensor<1x4096x576xbf16>
    %2248 = mhlo.copy %2247 : tensor<1x4096x576xbf16>
    %2249 = stablehlo.dot_general %2220, %2042#1, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x9x4096x4096xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<1x9x4096x128xbf16>
    %2250 = stablehlo.reshape %2249 : (tensor<1x9x4096x128xbf16>) -> tensor<1x9x4096x1x128xbf16>
    %2251 = stablehlo.transpose %2250, dims = [0, 2, 1, 3, 4] {result_layout = dense<[4, 3, 1, 2, 0]> : tensor<5xindex>, xla_shape = "bf16[1,4096,9,1,128]{4,3,1,2,0}"} : (tensor<1x9x4096x1x128xbf16>) -> tensor<1x4096x9x1x128xbf16>
    %2252 = stablehlo.reshape %2251 : (tensor<1x4096x9x1x128xbf16>) -> tensor<1x4096x9x128xbf16>
    %2253 = mhlo.copy %2252 : tensor<1x4096x9x128xbf16>
    %2254 = stablehlo.slice %2253 [0:1, 0:4096, 0:9, 64:128] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %2255 = stablehlo.multiply %2254, %2231 : tensor<1x4096x9x64xbf16>
    %2256 = stablehlo.slice %2253 [0:1, 0:4096, 0:9, 0:64] : (tensor<1x4096x9x128xbf16>) -> tensor<1x4096x9x64xbf16>
    %2257 = stablehlo.multiply %2256, %2237 : tensor<1x4096x9x64xbf16>
    %2258 = stablehlo.add %2255, %2257 : tensor<1x4096x9x64xbf16>
    %2259 = stablehlo.multiply %2254, %2237 : tensor<1x4096x9x64xbf16>
    %2260 = stablehlo.negate %2256 : tensor<1x4096x9x64xbf16>
    %2261 = stablehlo.multiply %2260, %2231 : tensor<1x4096x9x64xbf16>
    %2262 = stablehlo.add %2259, %2261 : tensor<1x4096x9x64xbf16>
    %2263 = stablehlo.concatenate %2258, %2262, dim = 3 : (tensor<1x4096x9x64xbf16>, tensor<1x4096x9x64xbf16>) -> tensor<1x4096x9x128xbf16>
    %2264 = "stablehlo.all_gather"(%2042#13) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 109, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> : (tensor<144x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2265 = stablehlo.dot_general %2263, %2264, contracting_dims = [2, 3] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<576x9x128xbf16>) -> tensor<1x4096x576xbf16>
    %2266 = mhlo.copy %2265 : tensor<1x4096x576xbf16>
    %2267 = stablehlo.add %2248, %2266 : tensor<1x4096x576xbf16>
    %2268 = mhlo.copy %2267 : tensor<1x4096x576xbf16>
    %2269 = stablehlo.multiply %2134, %2268 : tensor<1x4096x576xbf16>
    %2270 = stablehlo.reduce(%2269 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2271 = "stablehlo.all_reduce"(%2270) <{channel_handle = #stablehlo.channel_handle<handle = 110, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2272 = stablehlo.multiply %2271, %2271 : tensor<576xbf16>
    %2273 = stablehlo.convert %2272 : (tensor<576xbf16>) -> tensor<576xf32>
    %2274 = stablehlo.reduce(%2273 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2275 = stablehlo.convert %2274 : (tensor<f32>) -> tensor<bf16>
    %2276 = stablehlo.add %2123, %2275 : tensor<bf16>
    %2277 = stablehlo.broadcast_in_dim %2042#10, dims = [2] : (tensor<576xbf16>) -> tensor<1x4096x576xbf16>
    %2278 = stablehlo.multiply %2134, %2277 : tensor<1x4096x576xbf16>
    %2279 = mhlo.copy %2278 : tensor<1x4096x576xbf16>
    %2280 = mhlo.copy %2279 : tensor<1x4096x576xbf16>
    %2281 = stablehlo.dot_general %2244, %2280, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2282 = "stablehlo.all_reduce"(%2281) <{channel_handle = #stablehlo.channel_handle<handle = 111, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2283 = stablehlo.dynamic_slice %2282, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2284 = stablehlo.transpose %2283, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2285 = stablehlo.multiply %2284, %2284 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2286 = stablehlo.convert %2285 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2287 = stablehlo.reduce(%2286 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2288 = "stablehlo.all_reduce"(%2287) <{channel_handle = #stablehlo.channel_handle<handle = 112, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2289 = stablehlo.convert %2288 : (tensor<f32>) -> tensor<bf16>
    %2290 = stablehlo.add %2276, %2289 : tensor<bf16>
    %2291 = stablehlo.divide %2197, %2177 : tensor<1x4096x9x128xbf16>
    %2292 = mhlo.copy %2291 : tensor<1x4096x9x128xbf16>
    %2293 = stablehlo.dot_general %2151, %2292, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2294 = "stablehlo.all_reduce"(%2293) <{channel_handle = #stablehlo.channel_handle<handle = 113, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2295 = stablehlo.dynamic_slice %2294, %2080, %c_1, %c_1, sizes = [144, 9, 128] : (tensor<576x9x128xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<144x9x128xbf16>
    %2296 = stablehlo.transpose %2295, dims = [1, 2, 0] {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : (tensor<144x9x128xbf16>) -> tensor<9x128x144xbf16>
    %2297 = stablehlo.multiply %2296, %2296 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %2298 = stablehlo.convert %2297 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %2299 = stablehlo.reduce(%2298 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %2300 = "stablehlo.all_reduce"(%2299) <{channel_handle = #stablehlo.channel_handle<handle = 114, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2301 = stablehlo.convert %2300 : (tensor<f32>) -> tensor<bf16>
    %2302 = stablehlo.add %2290, %2301 : tensor<bf16>
    %2303 = stablehlo.dot_general %2263, %2280, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2304 = "stablehlo.all_reduce"(%2303) <{channel_handle = #stablehlo.channel_handle<handle = 115, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2305 = stablehlo.dynamic_slice %2304, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2306 = stablehlo.transpose %2305, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2307 = stablehlo.multiply %2306, %2306 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2308 = stablehlo.convert %2307 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2309 = stablehlo.reduce(%2308 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2310 = "stablehlo.all_reduce"(%2309) <{channel_handle = #stablehlo.channel_handle<handle = 116, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2311 = stablehlo.convert %2310 : (tensor<f32>) -> tensor<bf16>
    %2312 = stablehlo.add %2302, %2311 : tensor<bf16>
    %2313 = stablehlo.dot_general %2184, %2280, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2314 = "stablehlo.all_reduce"(%2313) <{channel_handle = #stablehlo.channel_handle<handle = 117, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2315 = stablehlo.dynamic_slice %2314, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2316 = stablehlo.transpose %2315, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2317 = stablehlo.multiply %2316, %2316 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2318 = stablehlo.convert %2317 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2319 = stablehlo.reduce(%2318 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2320 = "stablehlo.all_reduce"(%2319) <{channel_handle = #stablehlo.channel_handle<handle = 118, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2321 = stablehlo.convert %2320 : (tensor<f32>) -> tensor<bf16>
    %2322 = stablehlo.add %2312, %2321 : tensor<bf16>
    %2323 = stablehlo.multiply %1874, %1882 : tensor<1x4096x576xf32>
    %2324 = stablehlo.convert %2323 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2325 = stablehlo.multiply %2324, %1868 : tensor<1x4096x576xbf16>
    %2326 = mhlo.copy %2325 : tensor<1x4096x576xbf16>
    %2327 = stablehlo.dot_general %2326, %1863, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2328 = "stablehlo.all_reduce"(%2327) <{channel_handle = #stablehlo.channel_handle<handle = 119, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %2329 = stablehlo.dynamic_slice %2328, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2330 = stablehlo.multiply %2329, %2329 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2331 = stablehlo.convert %2330 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2332 = stablehlo.reduce(%2331 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2333 = "stablehlo.all_reduce"(%2332) <{channel_handle = #stablehlo.channel_handle<handle = 120, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2334 = stablehlo.convert %2333 : (tensor<f32>) -> tensor<bf16>
    %2335 = stablehlo.add %2322, %2334 : tensor<bf16>
    %2336 = stablehlo.dot_general %2326, %1854, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2337 = "stablehlo.all_reduce"(%2336) <{channel_handle = #stablehlo.channel_handle<handle = 121, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %2338 = stablehlo.dynamic_slice %2337, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2339 = stablehlo.multiply %2338, %2338 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2340 = stablehlo.convert %2339 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2341 = stablehlo.reduce(%2340 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2342 = "stablehlo.all_reduce"(%2341) <{channel_handle = #stablehlo.channel_handle<handle = 122, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2343 = stablehlo.convert %2342 : (tensor<f32>) -> tensor<bf16>
    %2344 = stablehlo.add %2335, %2343 : tensor<bf16>
    %2345 = stablehlo.multiply %1849, %1843#5 : tensor<1x4096x2016xbf16>
    %2346 = mhlo.copy %2345 : tensor<1x4096x2016xbf16>
    %2347 = stablehlo.dot_general %2346, %1850, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<1x4096x576xbf16>) -> tensor<2016x576xbf16>
    %2348 = "stablehlo.all_reduce"(%2347) <{channel_handle = #stablehlo.channel_handle<handle = 123, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<2016x576xbf16>) -> tensor<2016x576xbf16>
    %2349 = stablehlo.dynamic_slice %2348, %c_1, %2080, sizes = [2016, 144] : (tensor<2016x576xbf16>, tensor<i32>, tensor<i32>) -> tensor<2016x144xbf16>
    %2350 = stablehlo.multiply %2349, %2349 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %2351 = stablehlo.convert %2350 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %2352 = stablehlo.reduce(%2351 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %2353 = "stablehlo.all_reduce"(%2352) <{channel_handle = #stablehlo.channel_handle<handle = 124, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2354 = stablehlo.convert %2353 : (tensor<f32>) -> tensor<bf16>
    %2355 = stablehlo.add %2344, %2354 : tensor<bf16>
    %2356 = stablehlo.multiply %2324, %1867 : tensor<1x4096x576xbf16>
    %2357 = stablehlo.reduce(%2356 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2358 = "stablehlo.all_reduce"(%2357) <{channel_handle = #stablehlo.channel_handle<handle = 125, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2359 = stablehlo.multiply %2358, %2358 : tensor<576xbf16>
    %2360 = stablehlo.convert %2359 : (tensor<576xbf16>) -> tensor<576xf32>
    %2361 = stablehlo.reduce(%2360 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2362 = stablehlo.convert %2361 : (tensor<f32>) -> tensor<bf16>
    %2363 = stablehlo.add %2355, %2362 : tensor<bf16>
    %2364 = stablehlo.multiply %2018, %2026 : tensor<1x4096x576xf32>
    %2365 = stablehlo.convert %2364 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2366 = stablehlo.multiply %2365, %2014 : tensor<1x4096x576xbf16>
    %2367 = stablehlo.reduce(%2366 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2368 = "stablehlo.all_reduce"(%2367) <{channel_handle = #stablehlo.channel_handle<handle = 126, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2369 = stablehlo.multiply %2368, %2368 : tensor<576xbf16>
    %2370 = stablehlo.convert %2369 : (tensor<576xbf16>) -> tensor<576xf32>
    %2371 = stablehlo.reduce(%2370 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2372 = stablehlo.convert %2371 : (tensor<f32>) -> tensor<bf16>
    %2373 = stablehlo.add %2363, %2372 : tensor<bf16>
    %2374 = stablehlo.multiply %2365, %2015 : tensor<1x4096x576xbf16>
    %2375 = mhlo.copy %2374 : tensor<1x4096x576xbf16>
    %2376 = mhlo.copy %2375 : tensor<1x4096x576xbf16>
    %2377 = stablehlo.dot_general %1990, %2376, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2378 = "stablehlo.all_reduce"(%2377) <{channel_handle = #stablehlo.channel_handle<handle = 127, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2379 = stablehlo.dynamic_slice %2378, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2380 = stablehlo.transpose %2379, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2381 = stablehlo.multiply %2380, %2380 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2382 = stablehlo.convert %2381 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2383 = stablehlo.reduce(%2382 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2384 = "stablehlo.all_reduce"(%2383) <{channel_handle = #stablehlo.channel_handle<handle = 128, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2385 = stablehlo.convert %2384 : (tensor<f32>) -> tensor<bf16>
    %2386 = stablehlo.add %2373, %2385 : tensor<bf16>
    %2387 = stablehlo.divide %1943, %1923 : tensor<1x4096x9x128xbf16>
    %2388 = mhlo.copy %2387 : tensor<1x4096x9x128xbf16>
    %2389 = stablehlo.dot_general %1897, %2388, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2390 = "stablehlo.all_reduce"(%2389) <{channel_handle = #stablehlo.channel_handle<handle = 129, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2391 = stablehlo.dynamic_slice %2390, %2080, %c_1, %c_1, sizes = [144, 9, 128] : (tensor<576x9x128xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<144x9x128xbf16>
    %2392 = stablehlo.transpose %2391, dims = [1, 2, 0] {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : (tensor<144x9x128xbf16>) -> tensor<9x128x144xbf16>
    %2393 = stablehlo.multiply %2392, %2392 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %2394 = stablehlo.convert %2393 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %2395 = stablehlo.reduce(%2394 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %2396 = "stablehlo.all_reduce"(%2395) <{channel_handle = #stablehlo.channel_handle<handle = 130, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2397 = stablehlo.convert %2396 : (tensor<f32>) -> tensor<bf16>
    %2398 = stablehlo.add %2386, %2397 : tensor<bf16>
    %2399 = stablehlo.dot_general %2009, %2376, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2400 = "stablehlo.all_reduce"(%2399) <{channel_handle = #stablehlo.channel_handle<handle = 131, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2401 = stablehlo.dynamic_slice %2400, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2402 = stablehlo.transpose %2401, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2403 = stablehlo.multiply %2402, %2402 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2404 = stablehlo.convert %2403 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2405 = stablehlo.reduce(%2404 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2406 = "stablehlo.all_reduce"(%2405) <{channel_handle = #stablehlo.channel_handle<handle = 132, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2407 = stablehlo.convert %2406 : (tensor<f32>) -> tensor<bf16>
    %2408 = stablehlo.add %2398, %2407 : tensor<bf16>
    %2409 = stablehlo.dot_general %1930, %2376, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2410 = "stablehlo.all_reduce"(%2409) <{channel_handle = #stablehlo.channel_handle<handle = 133, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2411 = stablehlo.dynamic_slice %2410, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2412 = stablehlo.transpose %2411, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2413 = stablehlo.multiply %2412, %2412 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2414 = stablehlo.convert %2413 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2415 = stablehlo.reduce(%2414 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2416 = "stablehlo.all_reduce"(%2415) <{channel_handle = #stablehlo.channel_handle<handle = 134, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2417 = stablehlo.convert %2416 : (tensor<f32>) -> tensor<bf16>
    %2418 = stablehlo.add %2408, %2417 : tensor<bf16>
    %2419 = stablehlo.multiply %1675, %1683 : tensor<1x4096x576xf32>
    %2420 = stablehlo.convert %2419 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2421 = stablehlo.multiply %2420, %1669 : tensor<1x4096x576xbf16>
    %2422 = mhlo.copy %2421 : tensor<1x4096x576xbf16>
    %2423 = stablehlo.dot_general %2422, %1664, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2424 = "stablehlo.all_reduce"(%2423) <{channel_handle = #stablehlo.channel_handle<handle = 135, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %2425 = stablehlo.dynamic_slice %2424, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2426 = stablehlo.multiply %2425, %2425 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2427 = stablehlo.convert %2426 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2428 = stablehlo.reduce(%2427 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2429 = "stablehlo.all_reduce"(%2428) <{channel_handle = #stablehlo.channel_handle<handle = 136, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2430 = stablehlo.convert %2429 : (tensor<f32>) -> tensor<bf16>
    %2431 = stablehlo.add %2418, %2430 : tensor<bf16>
    %2432 = stablehlo.dot_general %2422, %1655, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2433 = "stablehlo.all_reduce"(%2432) <{channel_handle = #stablehlo.channel_handle<handle = 137, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %2434 = stablehlo.dynamic_slice %2433, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2435 = stablehlo.multiply %2434, %2434 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2436 = stablehlo.convert %2435 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2437 = stablehlo.reduce(%2436 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2438 = "stablehlo.all_reduce"(%2437) <{channel_handle = #stablehlo.channel_handle<handle = 138, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2439 = stablehlo.convert %2438 : (tensor<f32>) -> tensor<bf16>
    %2440 = stablehlo.add %2431, %2439 : tensor<bf16>
    %2441 = stablehlo.multiply %1650, %1644#5 : tensor<1x4096x2016xbf16>
    %2442 = mhlo.copy %2441 : tensor<1x4096x2016xbf16>
    %2443 = stablehlo.dot_general %2442, %1651, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<1x4096x576xbf16>) -> tensor<2016x576xbf16>
    %2444 = "stablehlo.all_reduce"(%2443) <{channel_handle = #stablehlo.channel_handle<handle = 139, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<2016x576xbf16>) -> tensor<2016x576xbf16>
    %2445 = stablehlo.dynamic_slice %2444, %c_1, %2080, sizes = [2016, 144] : (tensor<2016x576xbf16>, tensor<i32>, tensor<i32>) -> tensor<2016x144xbf16>
    %2446 = stablehlo.multiply %2445, %2445 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %2447 = stablehlo.convert %2446 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %2448 = stablehlo.reduce(%2447 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %2449 = "stablehlo.all_reduce"(%2448) <{channel_handle = #stablehlo.channel_handle<handle = 140, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2450 = stablehlo.convert %2449 : (tensor<f32>) -> tensor<bf16>
    %2451 = stablehlo.add %2440, %2450 : tensor<bf16>
    %2452 = stablehlo.multiply %2420, %1668 : tensor<1x4096x576xbf16>
    %2453 = stablehlo.reduce(%2452 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2454 = "stablehlo.all_reduce"(%2453) <{channel_handle = #stablehlo.channel_handle<handle = 141, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2455 = stablehlo.multiply %2454, %2454 : tensor<576xbf16>
    %2456 = stablehlo.convert %2455 : (tensor<576xbf16>) -> tensor<576xf32>
    %2457 = stablehlo.reduce(%2456 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2458 = stablehlo.convert %2457 : (tensor<f32>) -> tensor<bf16>
    %2459 = stablehlo.add %2451, %2458 : tensor<bf16>
    %2460 = stablehlo.multiply %1819, %1827 : tensor<1x4096x576xf32>
    %2461 = stablehlo.convert %2460 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2462 = stablehlo.multiply %2461, %1815 : tensor<1x4096x576xbf16>
    %2463 = stablehlo.reduce(%2462 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2464 = "stablehlo.all_reduce"(%2463) <{channel_handle = #stablehlo.channel_handle<handle = 142, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2465 = stablehlo.multiply %2464, %2464 : tensor<576xbf16>
    %2466 = stablehlo.convert %2465 : (tensor<576xbf16>) -> tensor<576xf32>
    %2467 = stablehlo.reduce(%2466 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2468 = stablehlo.convert %2467 : (tensor<f32>) -> tensor<bf16>
    %2469 = stablehlo.add %2459, %2468 : tensor<bf16>
    %2470 = stablehlo.multiply %2461, %1816 : tensor<1x4096x576xbf16>
    %2471 = mhlo.copy %2470 : tensor<1x4096x576xbf16>
    %2472 = mhlo.copy %2471 : tensor<1x4096x576xbf16>
    %2473 = stablehlo.dot_general %1791, %2472, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2474 = "stablehlo.all_reduce"(%2473) <{channel_handle = #stablehlo.channel_handle<handle = 143, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2475 = stablehlo.dynamic_slice %2474, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2476 = stablehlo.transpose %2475, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2477 = stablehlo.multiply %2476, %2476 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2478 = stablehlo.convert %2477 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2479 = stablehlo.reduce(%2478 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2480 = "stablehlo.all_reduce"(%2479) <{channel_handle = #stablehlo.channel_handle<handle = 144, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2481 = stablehlo.convert %2480 : (tensor<f32>) -> tensor<bf16>
    %2482 = stablehlo.add %2469, %2481 : tensor<bf16>
    %2483 = stablehlo.divide %1744, %1724 : tensor<1x4096x9x128xbf16>
    %2484 = mhlo.copy %2483 : tensor<1x4096x9x128xbf16>
    %2485 = stablehlo.dot_general %1698, %2484, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2486 = "stablehlo.all_reduce"(%2485) <{channel_handle = #stablehlo.channel_handle<handle = 145, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2487 = stablehlo.dynamic_slice %2486, %2080, %c_1, %c_1, sizes = [144, 9, 128] : (tensor<576x9x128xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<144x9x128xbf16>
    %2488 = stablehlo.transpose %2487, dims = [1, 2, 0] {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : (tensor<144x9x128xbf16>) -> tensor<9x128x144xbf16>
    %2489 = stablehlo.multiply %2488, %2488 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %2490 = stablehlo.convert %2489 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %2491 = stablehlo.reduce(%2490 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %2492 = "stablehlo.all_reduce"(%2491) <{channel_handle = #stablehlo.channel_handle<handle = 146, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2493 = stablehlo.convert %2492 : (tensor<f32>) -> tensor<bf16>
    %2494 = stablehlo.add %2482, %2493 : tensor<bf16>
    %2495 = stablehlo.dot_general %1810, %2472, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2496 = "stablehlo.all_reduce"(%2495) <{channel_handle = #stablehlo.channel_handle<handle = 147, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2497 = stablehlo.dynamic_slice %2496, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2498 = stablehlo.transpose %2497, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2499 = stablehlo.multiply %2498, %2498 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2500 = stablehlo.convert %2499 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2501 = stablehlo.reduce(%2500 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2502 = "stablehlo.all_reduce"(%2501) <{channel_handle = #stablehlo.channel_handle<handle = 148, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2503 = stablehlo.convert %2502 : (tensor<f32>) -> tensor<bf16>
    %2504 = stablehlo.add %2494, %2503 : tensor<bf16>
    %2505 = stablehlo.dot_general %1731, %2472, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2506 = "stablehlo.all_reduce"(%2505) <{channel_handle = #stablehlo.channel_handle<handle = 149, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2507 = stablehlo.dynamic_slice %2506, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2508 = stablehlo.transpose %2507, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2509 = stablehlo.multiply %2508, %2508 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2510 = stablehlo.convert %2509 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2511 = stablehlo.reduce(%2510 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2512 = "stablehlo.all_reduce"(%2511) <{channel_handle = #stablehlo.channel_handle<handle = 150, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2513 = stablehlo.convert %2512 : (tensor<f32>) -> tensor<bf16>
    %2514 = stablehlo.add %2504, %2513 : tensor<bf16>
    %2515 = stablehlo.multiply %1476, %1484 : tensor<1x4096x576xf32>
    %2516 = stablehlo.convert %2515 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2517 = stablehlo.multiply %2516, %1470 : tensor<1x4096x576xbf16>
    %2518 = mhlo.copy %2517 : tensor<1x4096x576xbf16>
    %2519 = stablehlo.dot_general %2518, %1465, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2520 = "stablehlo.all_reduce"(%2519) <{channel_handle = #stablehlo.channel_handle<handle = 151, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %2521 = stablehlo.dynamic_slice %2520, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2522 = stablehlo.multiply %2521, %2521 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2523 = stablehlo.convert %2522 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2524 = stablehlo.reduce(%2523 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2525 = "stablehlo.all_reduce"(%2524) <{channel_handle = #stablehlo.channel_handle<handle = 152, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2526 = stablehlo.convert %2525 : (tensor<f32>) -> tensor<bf16>
    %2527 = stablehlo.add %2514, %2526 : tensor<bf16>
    %2528 = stablehlo.dot_general %2518, %1456, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2529 = "stablehlo.all_reduce"(%2528) <{channel_handle = #stablehlo.channel_handle<handle = 153, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %2530 = stablehlo.dynamic_slice %2529, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2531 = stablehlo.multiply %2530, %2530 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2532 = stablehlo.convert %2531 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2533 = stablehlo.reduce(%2532 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2534 = "stablehlo.all_reduce"(%2533) <{channel_handle = #stablehlo.channel_handle<handle = 154, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2535 = stablehlo.convert %2534 : (tensor<f32>) -> tensor<bf16>
    %2536 = stablehlo.add %2527, %2535 : tensor<bf16>
    %2537 = stablehlo.multiply %1451, %1445#5 : tensor<1x4096x2016xbf16>
    %2538 = mhlo.copy %2537 : tensor<1x4096x2016xbf16>
    %2539 = stablehlo.dot_general %2538, %1452, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<1x4096x576xbf16>) -> tensor<2016x576xbf16>
    %2540 = "stablehlo.all_reduce"(%2539) <{channel_handle = #stablehlo.channel_handle<handle = 155, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<2016x576xbf16>) -> tensor<2016x576xbf16>
    %2541 = stablehlo.dynamic_slice %2540, %c_1, %2080, sizes = [2016, 144] : (tensor<2016x576xbf16>, tensor<i32>, tensor<i32>) -> tensor<2016x144xbf16>
    %2542 = stablehlo.multiply %2541, %2541 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %2543 = stablehlo.convert %2542 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %2544 = stablehlo.reduce(%2543 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %2545 = "stablehlo.all_reduce"(%2544) <{channel_handle = #stablehlo.channel_handle<handle = 156, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2546 = stablehlo.convert %2545 : (tensor<f32>) -> tensor<bf16>
    %2547 = stablehlo.add %2536, %2546 : tensor<bf16>
    %2548 = stablehlo.multiply %2516, %1469 : tensor<1x4096x576xbf16>
    %2549 = stablehlo.reduce(%2548 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2550 = "stablehlo.all_reduce"(%2549) <{channel_handle = #stablehlo.channel_handle<handle = 157, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2551 = stablehlo.multiply %2550, %2550 : tensor<576xbf16>
    %2552 = stablehlo.convert %2551 : (tensor<576xbf16>) -> tensor<576xf32>
    %2553 = stablehlo.reduce(%2552 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2554 = stablehlo.convert %2553 : (tensor<f32>) -> tensor<bf16>
    %2555 = stablehlo.add %2547, %2554 : tensor<bf16>
    %2556 = stablehlo.multiply %1620, %1628 : tensor<1x4096x576xf32>
    %2557 = stablehlo.convert %2556 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2558 = stablehlo.multiply %2557, %1616 : tensor<1x4096x576xbf16>
    %2559 = stablehlo.reduce(%2558 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2560 = "stablehlo.all_reduce"(%2559) <{channel_handle = #stablehlo.channel_handle<handle = 158, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2561 = stablehlo.multiply %2560, %2560 : tensor<576xbf16>
    %2562 = stablehlo.convert %2561 : (tensor<576xbf16>) -> tensor<576xf32>
    %2563 = stablehlo.reduce(%2562 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2564 = stablehlo.convert %2563 : (tensor<f32>) -> tensor<bf16>
    %2565 = stablehlo.add %2555, %2564 : tensor<bf16>
    %2566 = stablehlo.multiply %2557, %1617 : tensor<1x4096x576xbf16>
    %2567 = mhlo.copy %2566 : tensor<1x4096x576xbf16>
    %2568 = mhlo.copy %2567 : tensor<1x4096x576xbf16>
    %2569 = stablehlo.dot_general %1592, %2568, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2570 = "stablehlo.all_reduce"(%2569) <{channel_handle = #stablehlo.channel_handle<handle = 159, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2571 = stablehlo.dynamic_slice %2570, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2572 = stablehlo.transpose %2571, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2573 = stablehlo.multiply %2572, %2572 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2574 = stablehlo.convert %2573 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2575 = stablehlo.reduce(%2574 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2576 = "stablehlo.all_reduce"(%2575) <{channel_handle = #stablehlo.channel_handle<handle = 160, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2577 = stablehlo.convert %2576 : (tensor<f32>) -> tensor<bf16>
    %2578 = stablehlo.add %2565, %2577 : tensor<bf16>
    %2579 = stablehlo.divide %1545, %1525 : tensor<1x4096x9x128xbf16>
    %2580 = mhlo.copy %2579 : tensor<1x4096x9x128xbf16>
    %2581 = stablehlo.dot_general %1499, %2580, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2582 = "stablehlo.all_reduce"(%2581) <{channel_handle = #stablehlo.channel_handle<handle = 161, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2583 = stablehlo.dynamic_slice %2582, %2080, %c_1, %c_1, sizes = [144, 9, 128] : (tensor<576x9x128xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<144x9x128xbf16>
    %2584 = stablehlo.transpose %2583, dims = [1, 2, 0] {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : (tensor<144x9x128xbf16>) -> tensor<9x128x144xbf16>
    %2585 = stablehlo.multiply %2584, %2584 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %2586 = stablehlo.convert %2585 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %2587 = stablehlo.reduce(%2586 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %2588 = "stablehlo.all_reduce"(%2587) <{channel_handle = #stablehlo.channel_handle<handle = 162, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2589 = stablehlo.convert %2588 : (tensor<f32>) -> tensor<bf16>
    %2590 = stablehlo.add %2578, %2589 : tensor<bf16>
    %2591 = stablehlo.dot_general %1611, %2568, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2592 = "stablehlo.all_reduce"(%2591) <{channel_handle = #stablehlo.channel_handle<handle = 163, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2593 = stablehlo.dynamic_slice %2592, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2594 = stablehlo.transpose %2593, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2595 = stablehlo.multiply %2594, %2594 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2596 = stablehlo.convert %2595 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2597 = stablehlo.reduce(%2596 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2598 = "stablehlo.all_reduce"(%2597) <{channel_handle = #stablehlo.channel_handle<handle = 164, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2599 = stablehlo.convert %2598 : (tensor<f32>) -> tensor<bf16>
    %2600 = stablehlo.add %2590, %2599 : tensor<bf16>
    %2601 = stablehlo.dot_general %1532, %2568, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2602 = "stablehlo.all_reduce"(%2601) <{channel_handle = #stablehlo.channel_handle<handle = 165, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2603 = stablehlo.dynamic_slice %2602, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2604 = stablehlo.transpose %2603, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2605 = stablehlo.multiply %2604, %2604 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2606 = stablehlo.convert %2605 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2607 = stablehlo.reduce(%2606 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2608 = "stablehlo.all_reduce"(%2607) <{channel_handle = #stablehlo.channel_handle<handle = 166, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2609 = stablehlo.convert %2608 : (tensor<f32>) -> tensor<bf16>
    %2610 = stablehlo.add %2600, %2609 : tensor<bf16>
    %2611 = stablehlo.multiply %1277, %1285 : tensor<1x4096x576xf32>
    %2612 = stablehlo.convert %2611 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2613 = stablehlo.multiply %2612, %1271 : tensor<1x4096x576xbf16>
    %2614 = mhlo.copy %2613 : tensor<1x4096x576xbf16>
    %2615 = stablehlo.dot_general %2614, %1266, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2616 = "stablehlo.all_reduce"(%2615) <{channel_handle = #stablehlo.channel_handle<handle = 167, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %2617 = stablehlo.dynamic_slice %2616, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2618 = stablehlo.multiply %2617, %2617 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2619 = stablehlo.convert %2618 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2620 = stablehlo.reduce(%2619 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2621 = "stablehlo.all_reduce"(%2620) <{channel_handle = #stablehlo.channel_handle<handle = 168, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2622 = stablehlo.convert %2621 : (tensor<f32>) -> tensor<bf16>
    %2623 = stablehlo.add %2610, %2622 : tensor<bf16>
    %2624 = stablehlo.dot_general %2614, %1257, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2625 = "stablehlo.all_reduce"(%2624) <{channel_handle = #stablehlo.channel_handle<handle = 169, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %2626 = stablehlo.dynamic_slice %2625, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2627 = stablehlo.multiply %2626, %2626 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2628 = stablehlo.convert %2627 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2629 = stablehlo.reduce(%2628 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2630 = "stablehlo.all_reduce"(%2629) <{channel_handle = #stablehlo.channel_handle<handle = 170, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2631 = stablehlo.convert %2630 : (tensor<f32>) -> tensor<bf16>
    %2632 = stablehlo.add %2623, %2631 : tensor<bf16>
    %2633 = stablehlo.multiply %1252, %1246#5 : tensor<1x4096x2016xbf16>
    %2634 = mhlo.copy %2633 : tensor<1x4096x2016xbf16>
    %2635 = stablehlo.dot_general %2634, %1253, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<1x4096x576xbf16>) -> tensor<2016x576xbf16>
    %2636 = "stablehlo.all_reduce"(%2635) <{channel_handle = #stablehlo.channel_handle<handle = 171, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<2016x576xbf16>) -> tensor<2016x576xbf16>
    %2637 = stablehlo.dynamic_slice %2636, %c_1, %2080, sizes = [2016, 144] : (tensor<2016x576xbf16>, tensor<i32>, tensor<i32>) -> tensor<2016x144xbf16>
    %2638 = stablehlo.multiply %2637, %2637 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %2639 = stablehlo.convert %2638 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %2640 = stablehlo.reduce(%2639 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %2641 = "stablehlo.all_reduce"(%2640) <{channel_handle = #stablehlo.channel_handle<handle = 172, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2642 = stablehlo.convert %2641 : (tensor<f32>) -> tensor<bf16>
    %2643 = stablehlo.add %2632, %2642 : tensor<bf16>
    %2644 = stablehlo.multiply %2612, %1270 : tensor<1x4096x576xbf16>
    %2645 = stablehlo.reduce(%2644 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2646 = "stablehlo.all_reduce"(%2645) <{channel_handle = #stablehlo.channel_handle<handle = 173, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2647 = stablehlo.multiply %2646, %2646 : tensor<576xbf16>
    %2648 = stablehlo.convert %2647 : (tensor<576xbf16>) -> tensor<576xf32>
    %2649 = stablehlo.reduce(%2648 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2650 = stablehlo.convert %2649 : (tensor<f32>) -> tensor<bf16>
    %2651 = stablehlo.add %2643, %2650 : tensor<bf16>
    %2652 = stablehlo.multiply %1421, %1429 : tensor<1x4096x576xf32>
    %2653 = stablehlo.convert %2652 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2654 = stablehlo.multiply %2653, %1417 : tensor<1x4096x576xbf16>
    %2655 = stablehlo.reduce(%2654 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2656 = "stablehlo.all_reduce"(%2655) <{channel_handle = #stablehlo.channel_handle<handle = 174, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2657 = stablehlo.multiply %2656, %2656 : tensor<576xbf16>
    %2658 = stablehlo.convert %2657 : (tensor<576xbf16>) -> tensor<576xf32>
    %2659 = stablehlo.reduce(%2658 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2660 = stablehlo.convert %2659 : (tensor<f32>) -> tensor<bf16>
    %2661 = stablehlo.add %2651, %2660 : tensor<bf16>
    %2662 = stablehlo.multiply %2653, %1418 : tensor<1x4096x576xbf16>
    %2663 = mhlo.copy %2662 : tensor<1x4096x576xbf16>
    %2664 = mhlo.copy %2663 : tensor<1x4096x576xbf16>
    %2665 = stablehlo.dot_general %1393, %2664, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2666 = "stablehlo.all_reduce"(%2665) <{channel_handle = #stablehlo.channel_handle<handle = 175, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2667 = stablehlo.dynamic_slice %2666, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2668 = stablehlo.transpose %2667, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2669 = stablehlo.multiply %2668, %2668 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2670 = stablehlo.convert %2669 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2671 = stablehlo.reduce(%2670 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2672 = "stablehlo.all_reduce"(%2671) <{channel_handle = #stablehlo.channel_handle<handle = 176, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2673 = stablehlo.convert %2672 : (tensor<f32>) -> tensor<bf16>
    %2674 = stablehlo.add %2661, %2673 : tensor<bf16>
    %2675 = stablehlo.divide %1346, %1326 : tensor<1x4096x9x128xbf16>
    %2676 = mhlo.copy %2675 : tensor<1x4096x9x128xbf16>
    %2677 = stablehlo.dot_general %1300, %2676, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2678 = "stablehlo.all_reduce"(%2677) <{channel_handle = #stablehlo.channel_handle<handle = 177, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2679 = stablehlo.dynamic_slice %2678, %2080, %c_1, %c_1, sizes = [144, 9, 128] : (tensor<576x9x128xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<144x9x128xbf16>
    %2680 = stablehlo.transpose %2679, dims = [1, 2, 0] {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : (tensor<144x9x128xbf16>) -> tensor<9x128x144xbf16>
    %2681 = stablehlo.multiply %2680, %2680 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %2682 = stablehlo.convert %2681 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %2683 = stablehlo.reduce(%2682 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %2684 = "stablehlo.all_reduce"(%2683) <{channel_handle = #stablehlo.channel_handle<handle = 178, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2685 = stablehlo.convert %2684 : (tensor<f32>) -> tensor<bf16>
    %2686 = stablehlo.add %2674, %2685 : tensor<bf16>
    %2687 = stablehlo.dot_general %1412, %2664, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2688 = "stablehlo.all_reduce"(%2687) <{channel_handle = #stablehlo.channel_handle<handle = 179, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2689 = stablehlo.dynamic_slice %2688, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2690 = stablehlo.transpose %2689, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2691 = stablehlo.multiply %2690, %2690 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2692 = stablehlo.convert %2691 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2693 = stablehlo.reduce(%2692 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2694 = "stablehlo.all_reduce"(%2693) <{channel_handle = #stablehlo.channel_handle<handle = 180, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2695 = stablehlo.convert %2694 : (tensor<f32>) -> tensor<bf16>
    %2696 = stablehlo.add %2686, %2695 : tensor<bf16>
    %2697 = stablehlo.dot_general %1333, %2664, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2698 = "stablehlo.all_reduce"(%2697) <{channel_handle = #stablehlo.channel_handle<handle = 181, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2699 = stablehlo.dynamic_slice %2698, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2700 = stablehlo.transpose %2699, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2701 = stablehlo.multiply %2700, %2700 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2702 = stablehlo.convert %2701 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2703 = stablehlo.reduce(%2702 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2704 = "stablehlo.all_reduce"(%2703) <{channel_handle = #stablehlo.channel_handle<handle = 182, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2705 = stablehlo.convert %2704 : (tensor<f32>) -> tensor<bf16>
    %2706 = stablehlo.add %2696, %2705 : tensor<bf16>
    %2707 = stablehlo.multiply %1078, %1086 : tensor<1x4096x576xf32>
    %2708 = stablehlo.convert %2707 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2709 = stablehlo.multiply %2708, %1072 : tensor<1x4096x576xbf16>
    %2710 = mhlo.copy %2709 : tensor<1x4096x576xbf16>
    %2711 = stablehlo.dot_general %2710, %1067, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2712 = "stablehlo.all_reduce"(%2711) <{channel_handle = #stablehlo.channel_handle<handle = 183, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %2713 = stablehlo.dynamic_slice %2712, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2714 = stablehlo.multiply %2713, %2713 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2715 = stablehlo.convert %2714 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2716 = stablehlo.reduce(%2715 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2717 = "stablehlo.all_reduce"(%2716) <{channel_handle = #stablehlo.channel_handle<handle = 184, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2718 = stablehlo.convert %2717 : (tensor<f32>) -> tensor<bf16>
    %2719 = stablehlo.add %2706, %2718 : tensor<bf16>
    %2720 = stablehlo.dot_general %2710, %1058, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2721 = "stablehlo.all_reduce"(%2720) <{channel_handle = #stablehlo.channel_handle<handle = 185, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %2722 = stablehlo.dynamic_slice %2721, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2723 = stablehlo.multiply %2722, %2722 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2724 = stablehlo.convert %2723 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2725 = stablehlo.reduce(%2724 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2726 = "stablehlo.all_reduce"(%2725) <{channel_handle = #stablehlo.channel_handle<handle = 186, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2727 = stablehlo.convert %2726 : (tensor<f32>) -> tensor<bf16>
    %2728 = stablehlo.add %2719, %2727 : tensor<bf16>
    %2729 = stablehlo.multiply %1053, %1047#5 : tensor<1x4096x2016xbf16>
    %2730 = mhlo.copy %2729 : tensor<1x4096x2016xbf16>
    %2731 = stablehlo.dot_general %2730, %1054, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<1x4096x576xbf16>) -> tensor<2016x576xbf16>
    %2732 = "stablehlo.all_reduce"(%2731) <{channel_handle = #stablehlo.channel_handle<handle = 187, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<2016x576xbf16>) -> tensor<2016x576xbf16>
    %2733 = stablehlo.dynamic_slice %2732, %c_1, %2080, sizes = [2016, 144] : (tensor<2016x576xbf16>, tensor<i32>, tensor<i32>) -> tensor<2016x144xbf16>
    %2734 = stablehlo.multiply %2733, %2733 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %2735 = stablehlo.convert %2734 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %2736 = stablehlo.reduce(%2735 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %2737 = "stablehlo.all_reduce"(%2736) <{channel_handle = #stablehlo.channel_handle<handle = 188, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2738 = stablehlo.convert %2737 : (tensor<f32>) -> tensor<bf16>
    %2739 = stablehlo.add %2728, %2738 : tensor<bf16>
    %2740 = stablehlo.multiply %2708, %1071 : tensor<1x4096x576xbf16>
    %2741 = stablehlo.reduce(%2740 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2742 = "stablehlo.all_reduce"(%2741) <{channel_handle = #stablehlo.channel_handle<handle = 189, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2743 = stablehlo.multiply %2742, %2742 : tensor<576xbf16>
    %2744 = stablehlo.convert %2743 : (tensor<576xbf16>) -> tensor<576xf32>
    %2745 = stablehlo.reduce(%2744 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2746 = stablehlo.convert %2745 : (tensor<f32>) -> tensor<bf16>
    %2747 = stablehlo.add %2739, %2746 : tensor<bf16>
    %2748 = stablehlo.multiply %1222, %1230 : tensor<1x4096x576xf32>
    %2749 = stablehlo.convert %2748 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2750 = stablehlo.multiply %2749, %1218 : tensor<1x4096x576xbf16>
    %2751 = stablehlo.reduce(%2750 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2752 = "stablehlo.all_reduce"(%2751) <{channel_handle = #stablehlo.channel_handle<handle = 190, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2753 = stablehlo.multiply %2752, %2752 : tensor<576xbf16>
    %2754 = stablehlo.convert %2753 : (tensor<576xbf16>) -> tensor<576xf32>
    %2755 = stablehlo.reduce(%2754 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2756 = stablehlo.convert %2755 : (tensor<f32>) -> tensor<bf16>
    %2757 = stablehlo.add %2747, %2756 : tensor<bf16>
    %2758 = stablehlo.multiply %2749, %1219 : tensor<1x4096x576xbf16>
    %2759 = mhlo.copy %2758 : tensor<1x4096x576xbf16>
    %2760 = mhlo.copy %2759 : tensor<1x4096x576xbf16>
    %2761 = stablehlo.dot_general %1194, %2760, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2762 = "stablehlo.all_reduce"(%2761) <{channel_handle = #stablehlo.channel_handle<handle = 191, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2763 = stablehlo.dynamic_slice %2762, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2764 = stablehlo.transpose %2763, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2765 = stablehlo.multiply %2764, %2764 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2766 = stablehlo.convert %2765 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2767 = stablehlo.reduce(%2766 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2768 = "stablehlo.all_reduce"(%2767) <{channel_handle = #stablehlo.channel_handle<handle = 192, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2769 = stablehlo.convert %2768 : (tensor<f32>) -> tensor<bf16>
    %2770 = stablehlo.add %2757, %2769 : tensor<bf16>
    %2771 = stablehlo.divide %1147, %1127 : tensor<1x4096x9x128xbf16>
    %2772 = mhlo.copy %2771 : tensor<1x4096x9x128xbf16>
    %2773 = stablehlo.dot_general %1101, %2772, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2774 = "stablehlo.all_reduce"(%2773) <{channel_handle = #stablehlo.channel_handle<handle = 193, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2775 = stablehlo.dynamic_slice %2774, %2080, %c_1, %c_1, sizes = [144, 9, 128] : (tensor<576x9x128xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<144x9x128xbf16>
    %2776 = stablehlo.transpose %2775, dims = [1, 2, 0] {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : (tensor<144x9x128xbf16>) -> tensor<9x128x144xbf16>
    %2777 = stablehlo.multiply %2776, %2776 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %2778 = stablehlo.convert %2777 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %2779 = stablehlo.reduce(%2778 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %2780 = "stablehlo.all_reduce"(%2779) <{channel_handle = #stablehlo.channel_handle<handle = 194, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2781 = stablehlo.convert %2780 : (tensor<f32>) -> tensor<bf16>
    %2782 = stablehlo.add %2770, %2781 : tensor<bf16>
    %2783 = stablehlo.dot_general %1213, %2760, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2784 = "stablehlo.all_reduce"(%2783) <{channel_handle = #stablehlo.channel_handle<handle = 195, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2785 = stablehlo.dynamic_slice %2784, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2786 = stablehlo.transpose %2785, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2787 = stablehlo.multiply %2786, %2786 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2788 = stablehlo.convert %2787 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2789 = stablehlo.reduce(%2788 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2790 = "stablehlo.all_reduce"(%2789) <{channel_handle = #stablehlo.channel_handle<handle = 196, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2791 = stablehlo.convert %2790 : (tensor<f32>) -> tensor<bf16>
    %2792 = stablehlo.add %2782, %2791 : tensor<bf16>
    %2793 = stablehlo.dot_general %1134, %2760, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2794 = "stablehlo.all_reduce"(%2793) <{channel_handle = #stablehlo.channel_handle<handle = 197, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2795 = stablehlo.dynamic_slice %2794, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2796 = stablehlo.transpose %2795, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2797 = stablehlo.multiply %2796, %2796 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2798 = stablehlo.convert %2797 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2799 = stablehlo.reduce(%2798 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2800 = "stablehlo.all_reduce"(%2799) <{channel_handle = #stablehlo.channel_handle<handle = 198, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2801 = stablehlo.convert %2800 : (tensor<f32>) -> tensor<bf16>
    %2802 = stablehlo.add %2792, %2801 : tensor<bf16>
    %2803 = stablehlo.multiply %876, %884 : tensor<1x4096x576xf32>
    %2804 = stablehlo.convert %2803 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2805 = stablehlo.multiply %2804, %870 : tensor<1x4096x576xbf16>
    %2806 = mhlo.copy %2805 : tensor<1x4096x576xbf16>
    %2807 = stablehlo.dot_general %2806, %865, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2808 = "stablehlo.all_reduce"(%2807) <{channel_handle = #stablehlo.channel_handle<handle = 199, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %2809 = stablehlo.dynamic_slice %2808, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2810 = stablehlo.multiply %2809, %2809 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2811 = stablehlo.convert %2810 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2812 = stablehlo.reduce(%2811 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2813 = "stablehlo.all_reduce"(%2812) <{channel_handle = #stablehlo.channel_handle<handle = 200, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2814 = stablehlo.convert %2813 : (tensor<f32>) -> tensor<bf16>
    %2815 = stablehlo.add %2802, %2814 : tensor<bf16>
    %2816 = stablehlo.dot_general %2806, %856, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x2016xbf16>) -> tensor<576x2016xbf16>
    %2817 = "stablehlo.all_reduce"(%2816) <{channel_handle = #stablehlo.channel_handle<handle = 201, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x2016xbf16>) -> tensor<576x2016xbf16>
    %2818 = stablehlo.dynamic_slice %2817, %2080, %c_1, sizes = [144, 2016] : (tensor<576x2016xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x2016xbf16>
    %2819 = stablehlo.multiply %2818, %2818 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2820 = stablehlo.convert %2819 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %2821 = stablehlo.reduce(%2820 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %2822 = "stablehlo.all_reduce"(%2821) <{channel_handle = #stablehlo.channel_handle<handle = 202, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2823 = stablehlo.convert %2822 : (tensor<f32>) -> tensor<bf16>
    %2824 = stablehlo.add %2815, %2823 : tensor<bf16>
    %2825 = stablehlo.multiply %851, %845#5 : tensor<1x4096x2016xbf16>
    %2826 = mhlo.copy %2825 : tensor<1x4096x2016xbf16>
    %2827 = stablehlo.dot_general %2826, %852, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2016xbf16>, tensor<1x4096x576xbf16>) -> tensor<2016x576xbf16>
    %2828 = "stablehlo.all_reduce"(%2827) <{channel_handle = #stablehlo.channel_handle<handle = 203, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<2016x576xbf16>) -> tensor<2016x576xbf16>
    %2829 = stablehlo.dynamic_slice %2828, %c_1, %2080, sizes = [2016, 144] : (tensor<2016x576xbf16>, tensor<i32>, tensor<i32>) -> tensor<2016x144xbf16>
    %2830 = stablehlo.multiply %2829, %2829 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %2831 = stablehlo.convert %2830 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %2832 = stablehlo.reduce(%2831 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %2833 = "stablehlo.all_reduce"(%2832) <{channel_handle = #stablehlo.channel_handle<handle = 204, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2834 = stablehlo.convert %2833 : (tensor<f32>) -> tensor<bf16>
    %2835 = stablehlo.add %2824, %2834 : tensor<bf16>
    %2836 = stablehlo.multiply %2804, %869 : tensor<1x4096x576xbf16>
    %2837 = stablehlo.reduce(%2836 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2838 = "stablehlo.all_reduce"(%2837) <{channel_handle = #stablehlo.channel_handle<handle = 205, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2839 = stablehlo.multiply %2838, %2838 : tensor<576xbf16>
    %2840 = stablehlo.convert %2839 : (tensor<576xbf16>) -> tensor<576xf32>
    %2841 = stablehlo.reduce(%2840 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2842 = stablehlo.convert %2841 : (tensor<f32>) -> tensor<bf16>
    %2843 = stablehlo.add %2835, %2842 : tensor<bf16>
    %2844 = stablehlo.multiply %1023, %1031 : tensor<1x4096x576xf32>
    %2845 = stablehlo.convert %2844 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2846 = stablehlo.multiply %2845, %1019 : tensor<1x4096x576xbf16>
    %2847 = stablehlo.reduce(%2846 init: %cst_24) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096x576xbf16>, tensor<bf16>) -> tensor<576xbf16>
    %2848 = "stablehlo.all_reduce"(%2847) <{channel_handle = #stablehlo.channel_handle<handle = 206, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576xbf16>) -> tensor<576xbf16>
    %2849 = stablehlo.multiply %2848, %2848 : tensor<576xbf16>
    %2850 = stablehlo.convert %2849 : (tensor<576xbf16>) -> tensor<576xf32>
    %2851 = stablehlo.reduce(%2850 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %2852 = stablehlo.convert %2851 : (tensor<f32>) -> tensor<bf16>
    %2853 = stablehlo.add %2843, %2852 : tensor<bf16>
    %2854 = stablehlo.multiply %2845, %1020 : tensor<1x4096x576xbf16>
    %2855 = mhlo.copy %2854 : tensor<1x4096x576xbf16>
    %2856 = mhlo.copy %2855 : tensor<1x4096x576xbf16>
    %2857 = stablehlo.dot_general %995, %2856, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2858 = "stablehlo.all_reduce"(%2857) <{channel_handle = #stablehlo.channel_handle<handle = 207, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2859 = stablehlo.dynamic_slice %2858, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2860 = stablehlo.transpose %2859, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2861 = stablehlo.multiply %2860, %2860 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2862 = stablehlo.convert %2861 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2863 = stablehlo.reduce(%2862 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2864 = "stablehlo.all_reduce"(%2863) <{channel_handle = #stablehlo.channel_handle<handle = 208, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2865 = stablehlo.convert %2864 : (tensor<f32>) -> tensor<bf16>
    %2866 = stablehlo.add %2853, %2865 : tensor<bf16>
    %2867 = stablehlo.divide %946, %925 : tensor<1x4096x9x128xbf16>
    %2868 = mhlo.copy %2867 : tensor<1x4096x9x128xbf16>
    %2869 = stablehlo.dot_general %899, %2868, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2870 = "stablehlo.all_reduce"(%2869) <{channel_handle = #stablehlo.channel_handle<handle = 209, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x9x128xbf16>) -> tensor<576x9x128xbf16>
    %2871 = stablehlo.dynamic_slice %2870, %2080, %c_1, %c_1, sizes = [144, 9, 128] : (tensor<576x9x128xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<144x9x128xbf16>
    %2872 = stablehlo.transpose %2871, dims = [1, 2, 0] {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : (tensor<144x9x128xbf16>) -> tensor<9x128x144xbf16>
    %2873 = stablehlo.multiply %2872, %2872 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %2874 = stablehlo.convert %2873 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %2875 = stablehlo.reduce(%2874 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %2876 = "stablehlo.all_reduce"(%2875) <{channel_handle = #stablehlo.channel_handle<handle = 210, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2877 = stablehlo.convert %2876 : (tensor<f32>) -> tensor<bf16>
    %2878 = stablehlo.add %2866, %2877 : tensor<bf16>
    %2879 = stablehlo.dot_general %1014, %2856, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2880 = "stablehlo.all_reduce"(%2879) <{channel_handle = #stablehlo.channel_handle<handle = 211, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2881 = stablehlo.dynamic_slice %2880, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2882 = stablehlo.transpose %2881, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2883 = stablehlo.multiply %2882, %2882 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2884 = stablehlo.convert %2883 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2885 = stablehlo.reduce(%2884 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2886 = "stablehlo.all_reduce"(%2885) <{channel_handle = #stablehlo.channel_handle<handle = 212, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2887 = stablehlo.convert %2886 : (tensor<f32>) -> tensor<bf16>
    %2888 = stablehlo.add %2878, %2887 : tensor<bf16>
    %2889 = stablehlo.dot_general %932, %2856, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x9x128xbf16>, tensor<1x4096x576xbf16>) -> tensor<9x128x576xbf16>
    %2890 = "stablehlo.all_reduce"(%2889) <{channel_handle = #stablehlo.channel_handle<handle = 213, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<9x128x576xbf16>) -> tensor<9x128x576xbf16>
    %2891 = stablehlo.dynamic_slice %2890, %c_1, %c_1, %2080, sizes = [9, 128, 144] : (tensor<9x128x576xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<9x128x144xbf16>
    %2892 = stablehlo.transpose %2891, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : (tensor<9x128x144xbf16>) -> tensor<144x9x128xbf16>
    %2893 = stablehlo.multiply %2892, %2892 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %2894 = stablehlo.convert %2893 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %2895 = stablehlo.reduce(%2894 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %2896 = "stablehlo.all_reduce"(%2895) <{channel_handle = #stablehlo.channel_handle<handle = 214, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2897 = stablehlo.convert %2896 : (tensor<f32>) -> tensor<bf16>
    %2898 = stablehlo.add %2888, %2897 : tensor<bf16>
    %2899 = stablehlo.dot_general %793, %819, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x576xbf16>, tensor<1x4096x128256xbf16>) -> tensor<576x128256xbf16>
    %2900 = "stablehlo.all_reduce"(%2899) <{channel_handle = #stablehlo.channel_handle<handle = 215, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<576x128256xbf16>) -> tensor<576x128256xbf16>
    %2901 = stablehlo.dynamic_slice %2900, %2080, %c_1, sizes = [144, 128256] : (tensor<576x128256xbf16>, tensor<i32>, tensor<i32>) -> tensor<144x128256xbf16>
    %2902 = stablehlo.multiply %2901, %2901 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,128256]{0,1}"} : tensor<144x128256xbf16>
    %2903 = stablehlo.convert %2902 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,128256]{0,1}"} : (tensor<144x128256xbf16>) -> tensor<144x128256xf32>
    %2904 = stablehlo.reduce(%2903 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x128256xf32>, tensor<f32>) -> tensor<f32>
    %2905 = "stablehlo.all_reduce"(%2904) <{channel_handle = #stablehlo.channel_handle<handle = 216, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2906 = stablehlo.convert %2905 : (tensor<f32>) -> tensor<bf16>
    %2907 = stablehlo.add %2898, %2906 : tensor<bf16>
    %2908 = stablehlo.multiply %2268, %2277 : tensor<1x4096x576xbf16>
    %2909 = stablehlo.convert %2908 : (tensor<1x4096x576xbf16>) -> tensor<1x4096x576xf32>
    %2910 = stablehlo.multiply %2909, %2132 : tensor<1x4096x576xf32>
    %2911 = stablehlo.multiply %2124, %2909 : tensor<1x4096x576xf32>
    %2912 = stablehlo.reduce(%2911 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x576xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %2913 = stablehlo.reshape %2912 : (tensor<1x4096xf32>) -> tensor<1x4096x1xf32>
    %2914 = stablehlo.divide %2130, %2129 : tensor<1x4096x1xf32>
    %2915 = stablehlo.multiply %2914, %835 : tensor<1x4096x1xf32>
    %2916 = stablehlo.multiply %2913, %2915 : tensor<1x4096x1xf32>
    %2917 = stablehlo.multiply %2916, %838 : tensor<1x4096x1xf32>
    %2918 = stablehlo.reshape %2917 : (tensor<1x4096x1xf32>) -> tensor<1x4096xf32>
    %2919 = stablehlo.broadcast_in_dim %2918, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x576xf32>
    %2920 = stablehlo.multiply %2124, %2919 : tensor<1x4096x576xf32>
    %2921 = stablehlo.add %2910, %2920 : tensor<1x4096x576xf32>
    %2922 = stablehlo.convert %2921 : (tensor<1x4096x576xf32>) -> tensor<1x4096x576xbf16>
    %2923 = stablehlo.add %2150, %2922 : tensor<1x4096x576xbf16>
    %2924 = mhlo.copy %2923 : tensor<1x4096x576xbf16>
    %2925 = mhlo.copy %2924 : tensor<1x4096x576xbf16>
    %2926 = stablehlo.dot_general %26, %2925, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x128256xbf16>, tensor<1x4096x576xbf16>) -> tensor<128256x576xbf16>
    %2927 = "stablehlo.all_reduce"(%2926) <{channel_handle = #stablehlo.channel_handle<handle = 217, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<bf16>, %arg207: tensor<bf16>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<bf16>
      stablehlo.return %5085 : tensor<bf16>
    }) : (tensor<128256x576xbf16>) -> tensor<128256x576xbf16>
    %2928 = stablehlo.dynamic_slice %2927, %c_1, %2080, sizes = [128256, 144] : (tensor<128256x576xbf16>, tensor<i32>, tensor<i32>) -> tensor<128256x144xbf16>
    %2929 = stablehlo.multiply %2928, %2928 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128256,144]{0,1}"} : tensor<128256x144xbf16>
    %2930 = stablehlo.convert %2929 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128256,144]{0,1}"} : (tensor<128256x144xbf16>) -> tensor<128256x144xf32>
    %2931 = stablehlo.reduce(%2930 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<128256x144xf32>, tensor<f32>) -> tensor<f32>
    %2932 = "stablehlo.all_reduce"(%2931) <{channel_handle = #stablehlo.channel_handle<handle = 218, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %2933 = stablehlo.convert %2932 : (tensor<f32>) -> tensor<bf16>
    %2934 = stablehlo.add %2907, %2933 : tensor<bf16>
    %2935 = stablehlo.sqrt %2934 : tensor<bf16>
    %2936 = stablehlo.compare  LT, %2935, %cst_21 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
    %2937 = stablehlo.broadcast_in_dim %2936, dims = [] : (tensor<i1>) -> tensor<576xi1>
    %2938 = stablehlo.broadcast_in_dim %2935, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %2939 = stablehlo.divide %823, %2938 : tensor<576xbf16>
    %2940 = stablehlo.select %2937, %823, %2939 : tensor<576xi1>, tensor<576xbf16>
    %cst_28 = stablehlo.constant dense<1.000980e-01> : tensor<bf16>
    %2941 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %2942 = stablehlo.multiply %2940, %2941 : tensor<576xbf16>
    %cst_29 = stablehlo.constant dense<8.984370e-01> : tensor<bf16>
    %2943 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %2944 = stablehlo.multiply %arg68, %2943 : tensor<576xbf16>
    %2945 = stablehlo.add %2942, %2944 : tensor<576xbf16>
    %cst_30 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %c_31 = stablehlo.constant dense<2147483647> : tensor<i32>
    %2946 = stablehlo.compare  LT, %arg67, %c_31 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2947 = stablehlo.add %arg67, %c : tensor<i32>
    %2948 = stablehlo.select %2946, %2947, %c_31 : tensor<i1>, tensor<i32>
    %2949 = stablehlo.convert %2948 : (tensor<i32>) -> tensor<f32>
    %2950 = stablehlo.power %cst_30, %2949 : tensor<f32>
    %2951 = stablehlo.subtract %cst, %2950 : tensor<f32>
    %2952 = stablehlo.convert %2951 : (tensor<f32>) -> tensor<bf16>
    %2953 = stablehlo.broadcast_in_dim %2952, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %2954 = stablehlo.multiply %2940, %2940 : tensor<576xbf16>
    %cst_32 = stablehlo.constant dense<5.004880e-02> : tensor<bf16>
    %2955 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %2956 = stablehlo.multiply %2954, %2955 : tensor<576xbf16>
    %cst_33 = stablehlo.constant dense<9.492180e-01> : tensor<bf16>
    %2957 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %2958 = stablehlo.multiply %arg134, %2957 : tensor<576xbf16>
    %2959 = stablehlo.add %2956, %2958 : tensor<576xbf16>
    %cst_34 = stablehlo.constant dense<0.949999988> : tensor<f32>
    %2960 = stablehlo.power %cst_34, %2949 : tensor<f32>
    %2961 = stablehlo.subtract %cst, %2960 : tensor<f32>
    %2962 = stablehlo.convert %2961 : (tensor<f32>) -> tensor<bf16>
    %2963 = stablehlo.broadcast_in_dim %2962, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %2964 = stablehlo.divide %2959, %2963 : tensor<576xbf16>
    %2965 = stablehlo.sqrt %2964 : tensor<576xbf16>
    %cst_35 = stablehlo.constant dense<1.001170e-08> : tensor<bf16>
    %2966 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<bf16>) -> tensor<576xbf16>
    %2967 = stablehlo.add %2965, %2966 : tensor<576xbf16>
    %2968 = stablehlo.multiply %2953, %2967 : tensor<576xbf16>
    %2969 = stablehlo.divide %2945, %2968 : tensor<576xbf16>
    %2970 = stablehlo.multiply %arg1, %2941 : tensor<576xbf16>
    %2971 = stablehlo.add %2969, %2970 : tensor<576xbf16>
    %2972 = stablehlo.multiply %22, %2971 : tensor<576xbf16>
    %2973 = stablehlo.add %arg1, %2972 : tensor<576xbf16>
    %2974 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<144x2016xbf16>
    %2975 = stablehlo.broadcast_in_dim %2936, dims = [] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "pred[144,2016]{0,1}"} : (tensor<i1>) -> tensor<144x2016xi1>
    %2976 = stablehlo.broadcast_in_dim %2935, dims = [] : (tensor<bf16>) -> tensor<144x2016xbf16>
    %2977 = stablehlo.divide %2081, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2978 = stablehlo.select %2975, %2081, %2977 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %2979 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<bf16>) -> tensor<144x2016xbf16>
    %2980 = stablehlo.multiply %2978, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2981 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<bf16>) -> tensor<144x2016xbf16>
    %2982 = stablehlo.multiply %arg69, %2981 : tensor<144x2016xbf16>
    %2983 = stablehlo.add %2980, %2982 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2984 = stablehlo.broadcast_in_dim %2952, dims = [] : (tensor<bf16>) -> tensor<144x2016xbf16>
    %2985 = stablehlo.multiply %2978, %2978 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2986 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<bf16>) -> tensor<144x2016xbf16>
    %2987 = stablehlo.multiply %2985, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2988 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<bf16>) -> tensor<144x2016xbf16>
    %2989 = stablehlo.multiply %arg135, %2988 : tensor<144x2016xbf16>
    %2990 = stablehlo.add %2987, %2989 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2991 = stablehlo.broadcast_in_dim %2962, dims = [] : (tensor<bf16>) -> tensor<144x2016xbf16>
    %2992 = stablehlo.divide %2990, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2993 = stablehlo.sqrt %2992 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2994 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<bf16>) -> tensor<144x2016xbf16>
    %2995 = stablehlo.add %2993, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2996 = stablehlo.multiply %2984, %2995 : tensor<144x2016xbf16>
    %2997 = stablehlo.divide %2983, %2996 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %2998 = stablehlo.multiply %arg2, %2979 : tensor<144x2016xbf16>
    %2999 = stablehlo.add %2997, %2998 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3000 = stablehlo.multiply %2974, %2999 : tensor<144x2016xbf16>
    %3001 = stablehlo.add %arg2, %3000 : tensor<144x2016xbf16>
    %3002 = stablehlo.divide %2092, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3003 = stablehlo.select %2975, %2092, %3002 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %3004 = stablehlo.multiply %3003, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3005 = stablehlo.multiply %arg70, %2981 : tensor<144x2016xbf16>
    %3006 = stablehlo.add %3004, %3005 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3007 = stablehlo.multiply %3003, %3003 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3008 = stablehlo.multiply %3007, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3009 = stablehlo.multiply %arg136, %2988 : tensor<144x2016xbf16>
    %3010 = stablehlo.add %3008, %3009 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3011 = stablehlo.divide %3010, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3012 = stablehlo.sqrt %3011 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3013 = stablehlo.add %3012, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3014 = stablehlo.multiply %2984, %3013 : tensor<144x2016xbf16>
    %3015 = stablehlo.divide %3006, %3014 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3016 = stablehlo.multiply %arg3, %2979 : tensor<144x2016xbf16>
    %3017 = stablehlo.add %3015, %3016 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3018 = stablehlo.multiply %2974, %3017 : tensor<144x2016xbf16>
    %3019 = stablehlo.add %arg3, %3018 : tensor<144x2016xbf16>
    %3020 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<2016x144xbf16>
    %3021 = stablehlo.broadcast_in_dim %2936, dims = [] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "pred[2016,144]{0,1}"} : (tensor<i1>) -> tensor<2016x144xi1>
    %3022 = stablehlo.broadcast_in_dim %2935, dims = [] : (tensor<bf16>) -> tensor<2016x144xbf16>
    %3023 = stablehlo.divide %2103, %3022 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3024 = stablehlo.select %3021, %2103, %3023 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xi1>, tensor<2016x144xbf16>
    %3025 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<bf16>) -> tensor<2016x144xbf16>
    %3026 = stablehlo.multiply %3024, %3025 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3027 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<bf16>) -> tensor<2016x144xbf16>
    %3028 = stablehlo.multiply %arg71, %3027 : tensor<2016x144xbf16>
    %3029 = stablehlo.add %3026, %3028 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3030 = stablehlo.broadcast_in_dim %2952, dims = [] : (tensor<bf16>) -> tensor<2016x144xbf16>
    %3031 = stablehlo.multiply %3024, %3024 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3032 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<bf16>) -> tensor<2016x144xbf16>
    %3033 = stablehlo.multiply %3031, %3032 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3034 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<bf16>) -> tensor<2016x144xbf16>
    %3035 = stablehlo.multiply %arg137, %3034 : tensor<2016x144xbf16>
    %3036 = stablehlo.add %3033, %3035 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3037 = stablehlo.broadcast_in_dim %2962, dims = [] : (tensor<bf16>) -> tensor<2016x144xbf16>
    %3038 = stablehlo.divide %3036, %3037 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3039 = stablehlo.sqrt %3038 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3040 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<bf16>) -> tensor<2016x144xbf16>
    %3041 = stablehlo.add %3039, %3040 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3042 = stablehlo.multiply %3030, %3041 : tensor<2016x144xbf16>
    %3043 = stablehlo.divide %3029, %3042 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3044 = stablehlo.multiply %arg4, %3025 : tensor<2016x144xbf16>
    %3045 = stablehlo.add %3043, %3044 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3046 = stablehlo.multiply %3020, %3045 : tensor<2016x144xbf16>
    %3047 = stablehlo.add %arg4, %3046 : tensor<2016x144xbf16>
    %3048 = stablehlo.divide %2118, %2938 : tensor<576xbf16>
    %3049 = stablehlo.select %2937, %2118, %3048 : tensor<576xi1>, tensor<576xbf16>
    %3050 = stablehlo.multiply %3049, %2941 : tensor<576xbf16>
    %3051 = stablehlo.multiply %arg72, %2943 : tensor<576xbf16>
    %3052 = stablehlo.add %3050, %3051 : tensor<576xbf16>
    %3053 = stablehlo.multiply %3049, %3049 : tensor<576xbf16>
    %3054 = stablehlo.multiply %3053, %2955 : tensor<576xbf16>
    %3055 = stablehlo.multiply %arg138, %2957 : tensor<576xbf16>
    %3056 = stablehlo.add %3054, %3055 : tensor<576xbf16>
    %3057 = stablehlo.divide %3056, %2963 : tensor<576xbf16>
    %3058 = stablehlo.sqrt %3057 : tensor<576xbf16>
    %3059 = stablehlo.add %3058, %2966 : tensor<576xbf16>
    %3060 = stablehlo.multiply %2953, %3059 : tensor<576xbf16>
    %3061 = stablehlo.divide %3052, %3060 : tensor<576xbf16>
    %3062 = stablehlo.multiply %arg5, %2941 : tensor<576xbf16>
    %3063 = stablehlo.add %3061, %3062 : tensor<576xbf16>
    %3064 = stablehlo.multiply %22, %3063 : tensor<576xbf16>
    %3065 = stablehlo.add %arg5, %3064 : tensor<576xbf16>
    %3066 = stablehlo.divide %2271, %2938 : tensor<576xbf16>
    %3067 = stablehlo.select %2937, %2271, %3066 : tensor<576xi1>, tensor<576xbf16>
    %3068 = stablehlo.multiply %3067, %2941 : tensor<576xbf16>
    %3069 = stablehlo.multiply %arg73, %2943 : tensor<576xbf16>
    %3070 = stablehlo.add %3068, %3069 : tensor<576xbf16>
    %3071 = stablehlo.multiply %3067, %3067 : tensor<576xbf16>
    %3072 = stablehlo.multiply %3071, %2955 : tensor<576xbf16>
    %3073 = stablehlo.multiply %arg139, %2957 : tensor<576xbf16>
    %3074 = stablehlo.add %3072, %3073 : tensor<576xbf16>
    %3075 = stablehlo.divide %3074, %2963 : tensor<576xbf16>
    %3076 = stablehlo.sqrt %3075 : tensor<576xbf16>
    %3077 = stablehlo.add %3076, %2966 : tensor<576xbf16>
    %3078 = stablehlo.multiply %2953, %3077 : tensor<576xbf16>
    %3079 = stablehlo.divide %3070, %3078 : tensor<576xbf16>
    %3080 = stablehlo.multiply %arg6, %2941 : tensor<576xbf16>
    %3081 = stablehlo.add %3079, %3080 : tensor<576xbf16>
    %3082 = stablehlo.multiply %22, %3081 : tensor<576xbf16>
    %3083 = stablehlo.add %arg6, %3082 : tensor<576xbf16>
    %3084 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<144x9x128xbf16>
    %3085 = stablehlo.broadcast_in_dim %2936, dims = [] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "pred[144,9,128]{0,2,1}"} : (tensor<i1>) -> tensor<144x9x128xi1>
    %3086 = stablehlo.broadcast_in_dim %2935, dims = [] : (tensor<bf16>) -> tensor<144x9x128xbf16>
    %3087 = stablehlo.divide %2284, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3088 = stablehlo.select %3085, %2284, %3087 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3089 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<bf16>) -> tensor<144x9x128xbf16>
    %3090 = stablehlo.multiply %3088, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3091 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<bf16>) -> tensor<144x9x128xbf16>
    %3092 = stablehlo.multiply %arg74, %3091 : tensor<144x9x128xbf16>
    %3093 = stablehlo.add %3090, %3092 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3094 = stablehlo.broadcast_in_dim %2952, dims = [] : (tensor<bf16>) -> tensor<144x9x128xbf16>
    %3095 = stablehlo.multiply %3088, %3088 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3096 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<bf16>) -> tensor<144x9x128xbf16>
    %3097 = stablehlo.multiply %3095, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3098 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<bf16>) -> tensor<144x9x128xbf16>
    %3099 = stablehlo.multiply %arg140, %3098 : tensor<144x9x128xbf16>
    %3100 = stablehlo.add %3097, %3099 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3101 = stablehlo.broadcast_in_dim %2962, dims = [] : (tensor<bf16>) -> tensor<144x9x128xbf16>
    %3102 = stablehlo.divide %3100, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3103 = stablehlo.sqrt %3102 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3104 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<bf16>) -> tensor<144x9x128xbf16>
    %3105 = stablehlo.add %3103, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3106 = stablehlo.multiply %3094, %3105 : tensor<144x9x128xbf16>
    %3107 = stablehlo.divide %3093, %3106 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3108 = stablehlo.multiply %arg7, %3089 : tensor<144x9x128xbf16>
    %3109 = stablehlo.add %3107, %3108 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3110 = stablehlo.multiply %3084, %3109 : tensor<144x9x128xbf16>
    %3111 = stablehlo.add %arg7, %3110 : tensor<144x9x128xbf16>
    %3112 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<9x128x144xbf16>
    %3113 = stablehlo.broadcast_in_dim %2936, dims = [] {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "pred[9,128,144]{1,0,2}"} : (tensor<i1>) -> tensor<9x128x144xi1>
    %3114 = stablehlo.broadcast_in_dim %2935, dims = [] : (tensor<bf16>) -> tensor<9x128x144xbf16>
    %3115 = stablehlo.divide %2296, %3114 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3116 = stablehlo.select %3113, %2296, %3115 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xi1>, tensor<9x128x144xbf16>
    %3117 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<bf16>) -> tensor<9x128x144xbf16>
    %3118 = stablehlo.multiply %3116, %3117 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3119 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<bf16>) -> tensor<9x128x144xbf16>
    %3120 = stablehlo.multiply %arg75, %3119 : tensor<9x128x144xbf16>
    %3121 = stablehlo.add %3118, %3120 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3122 = stablehlo.broadcast_in_dim %2952, dims = [] : (tensor<bf16>) -> tensor<9x128x144xbf16>
    %3123 = stablehlo.multiply %3116, %3116 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3124 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<bf16>) -> tensor<9x128x144xbf16>
    %3125 = stablehlo.multiply %3123, %3124 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3126 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<bf16>) -> tensor<9x128x144xbf16>
    %3127 = stablehlo.multiply %arg141, %3126 : tensor<9x128x144xbf16>
    %3128 = stablehlo.add %3125, %3127 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3129 = stablehlo.broadcast_in_dim %2962, dims = [] : (tensor<bf16>) -> tensor<9x128x144xbf16>
    %3130 = stablehlo.divide %3128, %3129 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3131 = stablehlo.sqrt %3130 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3132 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<bf16>) -> tensor<9x128x144xbf16>
    %3133 = stablehlo.add %3131, %3132 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3134 = stablehlo.multiply %3122, %3133 : tensor<9x128x144xbf16>
    %3135 = stablehlo.divide %3121, %3134 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3136 = stablehlo.multiply %arg8, %3117 : tensor<9x128x144xbf16>
    %3137 = stablehlo.add %3135, %3136 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3138 = stablehlo.multiply %3112, %3137 : tensor<9x128x144xbf16>
    %3139 = stablehlo.add %arg8, %3138 : tensor<9x128x144xbf16>
    %3140 = stablehlo.divide %2306, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3141 = stablehlo.select %3085, %2306, %3140 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3142 = stablehlo.multiply %3141, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3143 = stablehlo.multiply %arg76, %3091 : tensor<144x9x128xbf16>
    %3144 = stablehlo.add %3142, %3143 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3145 = stablehlo.multiply %3141, %3141 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3146 = stablehlo.multiply %3145, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3147 = stablehlo.multiply %arg142, %3098 : tensor<144x9x128xbf16>
    %3148 = stablehlo.add %3146, %3147 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3149 = stablehlo.divide %3148, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3150 = stablehlo.sqrt %3149 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3151 = stablehlo.add %3150, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3152 = stablehlo.multiply %3094, %3151 : tensor<144x9x128xbf16>
    %3153 = stablehlo.divide %3144, %3152 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3154 = stablehlo.multiply %arg9, %3089 : tensor<144x9x128xbf16>
    %3155 = stablehlo.add %3153, %3154 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3156 = stablehlo.multiply %3084, %3155 : tensor<144x9x128xbf16>
    %3157 = stablehlo.add %arg9, %3156 : tensor<144x9x128xbf16>
    %3158 = stablehlo.divide %2316, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3159 = stablehlo.select %3085, %2316, %3158 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3160 = stablehlo.multiply %3159, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3161 = stablehlo.multiply %arg77, %3091 : tensor<144x9x128xbf16>
    %3162 = stablehlo.add %3160, %3161 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3163 = stablehlo.multiply %3159, %3159 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3164 = stablehlo.multiply %3163, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3165 = stablehlo.multiply %arg143, %3098 : tensor<144x9x128xbf16>
    %3166 = stablehlo.add %3164, %3165 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3167 = stablehlo.divide %3166, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3168 = stablehlo.sqrt %3167 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3169 = stablehlo.add %3168, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3170 = stablehlo.multiply %3094, %3169 : tensor<144x9x128xbf16>
    %3171 = stablehlo.divide %3162, %3170 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3172 = stablehlo.multiply %arg10, %3089 : tensor<144x9x128xbf16>
    %3173 = stablehlo.add %3171, %3172 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3174 = stablehlo.multiply %3084, %3173 : tensor<144x9x128xbf16>
    %3175 = stablehlo.add %arg10, %3174 : tensor<144x9x128xbf16>
    %3176 = stablehlo.divide %2329, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3177 = stablehlo.select %2975, %2329, %3176 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %3178 = stablehlo.multiply %3177, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3179 = stablehlo.multiply %arg78, %2981 : tensor<144x2016xbf16>
    %3180 = stablehlo.add %3178, %3179 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3181 = stablehlo.multiply %3177, %3177 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3182 = stablehlo.multiply %3181, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3183 = stablehlo.multiply %arg144, %2988 : tensor<144x2016xbf16>
    %3184 = stablehlo.add %3182, %3183 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3185 = stablehlo.divide %3184, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3186 = stablehlo.sqrt %3185 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3187 = stablehlo.add %3186, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3188 = stablehlo.multiply %2984, %3187 : tensor<144x2016xbf16>
    %3189 = stablehlo.divide %3180, %3188 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3190 = stablehlo.multiply %arg11, %2979 : tensor<144x2016xbf16>
    %3191 = stablehlo.add %3189, %3190 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3192 = stablehlo.multiply %2974, %3191 : tensor<144x2016xbf16>
    %3193 = stablehlo.add %arg11, %3192 : tensor<144x2016xbf16>
    %3194 = stablehlo.divide %2338, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3195 = stablehlo.select %2975, %2338, %3194 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %3196 = stablehlo.multiply %3195, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3197 = stablehlo.multiply %arg79, %2981 : tensor<144x2016xbf16>
    %3198 = stablehlo.add %3196, %3197 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3199 = stablehlo.multiply %3195, %3195 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3200 = stablehlo.multiply %3199, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3201 = stablehlo.multiply %arg145, %2988 : tensor<144x2016xbf16>
    %3202 = stablehlo.add %3200, %3201 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3203 = stablehlo.divide %3202, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3204 = stablehlo.sqrt %3203 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3205 = stablehlo.add %3204, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3206 = stablehlo.multiply %2984, %3205 : tensor<144x2016xbf16>
    %3207 = stablehlo.divide %3198, %3206 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3208 = stablehlo.multiply %arg12, %2979 : tensor<144x2016xbf16>
    %3209 = stablehlo.add %3207, %3208 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3210 = stablehlo.multiply %2974, %3209 : tensor<144x2016xbf16>
    %3211 = stablehlo.add %arg12, %3210 : tensor<144x2016xbf16>
    %3212 = stablehlo.divide %2349, %3022 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3213 = stablehlo.select %3021, %2349, %3212 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xi1>, tensor<2016x144xbf16>
    %3214 = stablehlo.multiply %3213, %3025 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3215 = stablehlo.multiply %arg80, %3027 : tensor<2016x144xbf16>
    %3216 = stablehlo.add %3214, %3215 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3217 = stablehlo.multiply %3213, %3213 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3218 = stablehlo.multiply %3217, %3032 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3219 = stablehlo.multiply %arg146, %3034 : tensor<2016x144xbf16>
    %3220 = stablehlo.add %3218, %3219 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3221 = stablehlo.divide %3220, %3037 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3222 = stablehlo.sqrt %3221 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3223 = stablehlo.add %3222, %3040 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3224 = stablehlo.multiply %3030, %3223 : tensor<2016x144xbf16>
    %3225 = stablehlo.divide %3216, %3224 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3226 = stablehlo.multiply %arg13, %3025 : tensor<2016x144xbf16>
    %3227 = stablehlo.add %3225, %3226 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3228 = stablehlo.multiply %3020, %3227 : tensor<2016x144xbf16>
    %3229 = stablehlo.add %arg13, %3228 : tensor<2016x144xbf16>
    %3230 = stablehlo.divide %2358, %2938 : tensor<576xbf16>
    %3231 = stablehlo.select %2937, %2358, %3230 : tensor<576xi1>, tensor<576xbf16>
    %3232 = stablehlo.multiply %3231, %2941 : tensor<576xbf16>
    %3233 = stablehlo.multiply %arg81, %2943 : tensor<576xbf16>
    %3234 = stablehlo.add %3232, %3233 : tensor<576xbf16>
    %3235 = stablehlo.multiply %3231, %3231 : tensor<576xbf16>
    %3236 = stablehlo.multiply %3235, %2955 : tensor<576xbf16>
    %3237 = stablehlo.multiply %arg147, %2957 : tensor<576xbf16>
    %3238 = stablehlo.add %3236, %3237 : tensor<576xbf16>
    %3239 = stablehlo.divide %3238, %2963 : tensor<576xbf16>
    %3240 = stablehlo.sqrt %3239 : tensor<576xbf16>
    %3241 = stablehlo.add %3240, %2966 : tensor<576xbf16>
    %3242 = stablehlo.multiply %2953, %3241 : tensor<576xbf16>
    %3243 = stablehlo.divide %3234, %3242 : tensor<576xbf16>
    %3244 = stablehlo.multiply %arg14, %2941 : tensor<576xbf16>
    %3245 = stablehlo.add %3243, %3244 : tensor<576xbf16>
    %3246 = stablehlo.multiply %22, %3245 : tensor<576xbf16>
    %3247 = stablehlo.add %arg14, %3246 : tensor<576xbf16>
    %3248 = stablehlo.divide %2368, %2938 : tensor<576xbf16>
    %3249 = stablehlo.select %2937, %2368, %3248 : tensor<576xi1>, tensor<576xbf16>
    %3250 = stablehlo.multiply %3249, %2941 : tensor<576xbf16>
    %3251 = stablehlo.multiply %arg82, %2943 : tensor<576xbf16>
    %3252 = stablehlo.add %3250, %3251 : tensor<576xbf16>
    %3253 = stablehlo.multiply %3249, %3249 : tensor<576xbf16>
    %3254 = stablehlo.multiply %3253, %2955 : tensor<576xbf16>
    %3255 = stablehlo.multiply %arg148, %2957 : tensor<576xbf16>
    %3256 = stablehlo.add %3254, %3255 : tensor<576xbf16>
    %3257 = stablehlo.divide %3256, %2963 : tensor<576xbf16>
    %3258 = stablehlo.sqrt %3257 : tensor<576xbf16>
    %3259 = stablehlo.add %3258, %2966 : tensor<576xbf16>
    %3260 = stablehlo.multiply %2953, %3259 : tensor<576xbf16>
    %3261 = stablehlo.divide %3252, %3260 : tensor<576xbf16>
    %3262 = stablehlo.multiply %arg15, %2941 : tensor<576xbf16>
    %3263 = stablehlo.add %3261, %3262 : tensor<576xbf16>
    %3264 = stablehlo.multiply %22, %3263 : tensor<576xbf16>
    %3265 = stablehlo.add %arg15, %3264 : tensor<576xbf16>
    %3266 = stablehlo.divide %2380, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3267 = stablehlo.select %3085, %2380, %3266 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3268 = stablehlo.multiply %3267, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3269 = stablehlo.multiply %arg83, %3091 : tensor<144x9x128xbf16>
    %3270 = stablehlo.add %3268, %3269 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3271 = stablehlo.multiply %3267, %3267 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3272 = stablehlo.multiply %3271, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3273 = stablehlo.multiply %arg149, %3098 : tensor<144x9x128xbf16>
    %3274 = stablehlo.add %3272, %3273 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3275 = stablehlo.divide %3274, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3276 = stablehlo.sqrt %3275 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3277 = stablehlo.add %3276, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3278 = stablehlo.multiply %3094, %3277 : tensor<144x9x128xbf16>
    %3279 = stablehlo.divide %3270, %3278 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3280 = stablehlo.multiply %arg16, %3089 : tensor<144x9x128xbf16>
    %3281 = stablehlo.add %3279, %3280 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3282 = stablehlo.multiply %3084, %3281 : tensor<144x9x128xbf16>
    %3283 = stablehlo.add %arg16, %3282 : tensor<144x9x128xbf16>
    %3284 = stablehlo.divide %2392, %3114 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3285 = stablehlo.select %3113, %2392, %3284 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xi1>, tensor<9x128x144xbf16>
    %3286 = stablehlo.multiply %3285, %3117 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3287 = stablehlo.multiply %arg84, %3119 : tensor<9x128x144xbf16>
    %3288 = stablehlo.add %3286, %3287 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3289 = stablehlo.multiply %3285, %3285 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3290 = stablehlo.multiply %3289, %3124 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3291 = stablehlo.multiply %arg150, %3126 : tensor<9x128x144xbf16>
    %3292 = stablehlo.add %3290, %3291 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3293 = stablehlo.divide %3292, %3129 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3294 = stablehlo.sqrt %3293 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3295 = stablehlo.add %3294, %3132 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3296 = stablehlo.multiply %3122, %3295 : tensor<9x128x144xbf16>
    %3297 = stablehlo.divide %3288, %3296 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3298 = stablehlo.multiply %arg17, %3117 : tensor<9x128x144xbf16>
    %3299 = stablehlo.add %3297, %3298 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3300 = stablehlo.multiply %3112, %3299 : tensor<9x128x144xbf16>
    %3301 = stablehlo.add %arg17, %3300 : tensor<9x128x144xbf16>
    %3302 = stablehlo.divide %2402, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3303 = stablehlo.select %3085, %2402, %3302 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3304 = stablehlo.multiply %3303, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3305 = stablehlo.multiply %arg85, %3091 : tensor<144x9x128xbf16>
    %3306 = stablehlo.add %3304, %3305 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3307 = stablehlo.multiply %3303, %3303 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3308 = stablehlo.multiply %3307, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3309 = stablehlo.multiply %arg151, %3098 : tensor<144x9x128xbf16>
    %3310 = stablehlo.add %3308, %3309 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3311 = stablehlo.divide %3310, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3312 = stablehlo.sqrt %3311 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3313 = stablehlo.add %3312, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3314 = stablehlo.multiply %3094, %3313 : tensor<144x9x128xbf16>
    %3315 = stablehlo.divide %3306, %3314 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3316 = stablehlo.multiply %arg18, %3089 : tensor<144x9x128xbf16>
    %3317 = stablehlo.add %3315, %3316 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3318 = stablehlo.multiply %3084, %3317 : tensor<144x9x128xbf16>
    %3319 = stablehlo.add %arg18, %3318 : tensor<144x9x128xbf16>
    %3320 = stablehlo.divide %2412, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3321 = stablehlo.select %3085, %2412, %3320 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3322 = stablehlo.multiply %3321, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3323 = stablehlo.multiply %arg86, %3091 : tensor<144x9x128xbf16>
    %3324 = stablehlo.add %3322, %3323 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3325 = stablehlo.multiply %3321, %3321 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3326 = stablehlo.multiply %3325, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3327 = stablehlo.multiply %arg152, %3098 : tensor<144x9x128xbf16>
    %3328 = stablehlo.add %3326, %3327 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3329 = stablehlo.divide %3328, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3330 = stablehlo.sqrt %3329 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3331 = stablehlo.add %3330, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3332 = stablehlo.multiply %3094, %3331 : tensor<144x9x128xbf16>
    %3333 = stablehlo.divide %3324, %3332 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3334 = stablehlo.multiply %arg19, %3089 : tensor<144x9x128xbf16>
    %3335 = stablehlo.add %3333, %3334 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3336 = stablehlo.multiply %3084, %3335 : tensor<144x9x128xbf16>
    %3337 = stablehlo.add %arg19, %3336 : tensor<144x9x128xbf16>
    %3338 = stablehlo.divide %2425, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3339 = stablehlo.select %2975, %2425, %3338 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %3340 = stablehlo.multiply %3339, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3341 = stablehlo.multiply %arg87, %2981 : tensor<144x2016xbf16>
    %3342 = stablehlo.add %3340, %3341 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3343 = stablehlo.multiply %3339, %3339 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3344 = stablehlo.multiply %3343, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3345 = stablehlo.multiply %arg153, %2988 : tensor<144x2016xbf16>
    %3346 = stablehlo.add %3344, %3345 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3347 = stablehlo.divide %3346, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3348 = stablehlo.sqrt %3347 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3349 = stablehlo.add %3348, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3350 = stablehlo.multiply %2984, %3349 : tensor<144x2016xbf16>
    %3351 = stablehlo.divide %3342, %3350 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3352 = stablehlo.multiply %arg20, %2979 : tensor<144x2016xbf16>
    %3353 = stablehlo.add %3351, %3352 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3354 = stablehlo.multiply %2974, %3353 : tensor<144x2016xbf16>
    %3355 = stablehlo.add %arg20, %3354 : tensor<144x2016xbf16>
    %3356 = stablehlo.divide %2434, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3357 = stablehlo.select %2975, %2434, %3356 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %3358 = stablehlo.multiply %3357, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3359 = stablehlo.multiply %arg88, %2981 : tensor<144x2016xbf16>
    %3360 = stablehlo.add %3358, %3359 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3361 = stablehlo.multiply %3357, %3357 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3362 = stablehlo.multiply %3361, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3363 = stablehlo.multiply %arg154, %2988 : tensor<144x2016xbf16>
    %3364 = stablehlo.add %3362, %3363 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3365 = stablehlo.divide %3364, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3366 = stablehlo.sqrt %3365 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3367 = stablehlo.add %3366, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3368 = stablehlo.multiply %2984, %3367 : tensor<144x2016xbf16>
    %3369 = stablehlo.divide %3360, %3368 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3370 = stablehlo.multiply %arg21, %2979 : tensor<144x2016xbf16>
    %3371 = stablehlo.add %3369, %3370 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3372 = stablehlo.multiply %2974, %3371 : tensor<144x2016xbf16>
    %3373 = stablehlo.add %arg21, %3372 : tensor<144x2016xbf16>
    %3374 = stablehlo.divide %2445, %3022 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3375 = stablehlo.select %3021, %2445, %3374 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xi1>, tensor<2016x144xbf16>
    %3376 = stablehlo.multiply %3375, %3025 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3377 = stablehlo.multiply %arg89, %3027 : tensor<2016x144xbf16>
    %3378 = stablehlo.add %3376, %3377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3379 = stablehlo.multiply %3375, %3375 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3380 = stablehlo.multiply %3379, %3032 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3381 = stablehlo.multiply %arg155, %3034 : tensor<2016x144xbf16>
    %3382 = stablehlo.add %3380, %3381 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3383 = stablehlo.divide %3382, %3037 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3384 = stablehlo.sqrt %3383 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3385 = stablehlo.add %3384, %3040 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3386 = stablehlo.multiply %3030, %3385 : tensor<2016x144xbf16>
    %3387 = stablehlo.divide %3378, %3386 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3388 = stablehlo.multiply %arg22, %3025 : tensor<2016x144xbf16>
    %3389 = stablehlo.add %3387, %3388 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3390 = stablehlo.multiply %3020, %3389 : tensor<2016x144xbf16>
    %3391 = stablehlo.add %arg22, %3390 : tensor<2016x144xbf16>
    %3392 = stablehlo.divide %2454, %2938 : tensor<576xbf16>
    %3393 = stablehlo.select %2937, %2454, %3392 : tensor<576xi1>, tensor<576xbf16>
    %3394 = stablehlo.multiply %3393, %2941 : tensor<576xbf16>
    %3395 = stablehlo.multiply %arg90, %2943 : tensor<576xbf16>
    %3396 = stablehlo.add %3394, %3395 : tensor<576xbf16>
    %3397 = stablehlo.multiply %3393, %3393 : tensor<576xbf16>
    %3398 = stablehlo.multiply %3397, %2955 : tensor<576xbf16>
    %3399 = stablehlo.multiply %arg156, %2957 : tensor<576xbf16>
    %3400 = stablehlo.add %3398, %3399 : tensor<576xbf16>
    %3401 = stablehlo.divide %3400, %2963 : tensor<576xbf16>
    %3402 = stablehlo.sqrt %3401 : tensor<576xbf16>
    %3403 = stablehlo.add %3402, %2966 : tensor<576xbf16>
    %3404 = stablehlo.multiply %2953, %3403 : tensor<576xbf16>
    %3405 = stablehlo.divide %3396, %3404 : tensor<576xbf16>
    %3406 = stablehlo.multiply %arg23, %2941 : tensor<576xbf16>
    %3407 = stablehlo.add %3405, %3406 : tensor<576xbf16>
    %3408 = stablehlo.multiply %22, %3407 : tensor<576xbf16>
    %3409 = stablehlo.add %arg23, %3408 : tensor<576xbf16>
    %3410 = stablehlo.divide %2464, %2938 : tensor<576xbf16>
    %3411 = stablehlo.select %2937, %2464, %3410 : tensor<576xi1>, tensor<576xbf16>
    %3412 = stablehlo.multiply %3411, %2941 : tensor<576xbf16>
    %3413 = stablehlo.multiply %arg91, %2943 : tensor<576xbf16>
    %3414 = stablehlo.add %3412, %3413 : tensor<576xbf16>
    %3415 = stablehlo.multiply %3411, %3411 : tensor<576xbf16>
    %3416 = stablehlo.multiply %3415, %2955 : tensor<576xbf16>
    %3417 = stablehlo.multiply %arg157, %2957 : tensor<576xbf16>
    %3418 = stablehlo.add %3416, %3417 : tensor<576xbf16>
    %3419 = stablehlo.divide %3418, %2963 : tensor<576xbf16>
    %3420 = stablehlo.sqrt %3419 : tensor<576xbf16>
    %3421 = stablehlo.add %3420, %2966 : tensor<576xbf16>
    %3422 = stablehlo.multiply %2953, %3421 : tensor<576xbf16>
    %3423 = stablehlo.divide %3414, %3422 : tensor<576xbf16>
    %3424 = stablehlo.multiply %arg24, %2941 : tensor<576xbf16>
    %3425 = stablehlo.add %3423, %3424 : tensor<576xbf16>
    %3426 = stablehlo.multiply %22, %3425 : tensor<576xbf16>
    %3427 = stablehlo.add %arg24, %3426 : tensor<576xbf16>
    %3428 = stablehlo.divide %2476, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3429 = stablehlo.select %3085, %2476, %3428 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3430 = stablehlo.multiply %3429, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3431 = stablehlo.multiply %arg92, %3091 : tensor<144x9x128xbf16>
    %3432 = stablehlo.add %3430, %3431 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3433 = stablehlo.multiply %3429, %3429 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3434 = stablehlo.multiply %3433, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3435 = stablehlo.multiply %arg158, %3098 : tensor<144x9x128xbf16>
    %3436 = stablehlo.add %3434, %3435 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3437 = stablehlo.divide %3436, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3438 = stablehlo.sqrt %3437 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3439 = stablehlo.add %3438, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3440 = stablehlo.multiply %3094, %3439 : tensor<144x9x128xbf16>
    %3441 = stablehlo.divide %3432, %3440 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3442 = stablehlo.multiply %arg25, %3089 : tensor<144x9x128xbf16>
    %3443 = stablehlo.add %3441, %3442 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3444 = stablehlo.multiply %3084, %3443 : tensor<144x9x128xbf16>
    %3445 = stablehlo.add %arg25, %3444 : tensor<144x9x128xbf16>
    %3446 = stablehlo.divide %2488, %3114 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3447 = stablehlo.select %3113, %2488, %3446 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xi1>, tensor<9x128x144xbf16>
    %3448 = stablehlo.multiply %3447, %3117 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3449 = stablehlo.multiply %arg93, %3119 : tensor<9x128x144xbf16>
    %3450 = stablehlo.add %3448, %3449 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3451 = stablehlo.multiply %3447, %3447 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3452 = stablehlo.multiply %3451, %3124 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3453 = stablehlo.multiply %arg159, %3126 : tensor<9x128x144xbf16>
    %3454 = stablehlo.add %3452, %3453 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3455 = stablehlo.divide %3454, %3129 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3456 = stablehlo.sqrt %3455 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3457 = stablehlo.add %3456, %3132 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3458 = stablehlo.multiply %3122, %3457 : tensor<9x128x144xbf16>
    %3459 = stablehlo.divide %3450, %3458 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3460 = stablehlo.multiply %arg26, %3117 : tensor<9x128x144xbf16>
    %3461 = stablehlo.add %3459, %3460 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3462 = stablehlo.multiply %3112, %3461 : tensor<9x128x144xbf16>
    %3463 = stablehlo.add %arg26, %3462 : tensor<9x128x144xbf16>
    %3464 = stablehlo.divide %2498, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3465 = stablehlo.select %3085, %2498, %3464 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3466 = stablehlo.multiply %3465, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3467 = stablehlo.multiply %arg94, %3091 : tensor<144x9x128xbf16>
    %3468 = stablehlo.add %3466, %3467 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3469 = stablehlo.multiply %3465, %3465 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3470 = stablehlo.multiply %3469, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3471 = stablehlo.multiply %arg160, %3098 : tensor<144x9x128xbf16>
    %3472 = stablehlo.add %3470, %3471 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3473 = stablehlo.divide %3472, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3474 = stablehlo.sqrt %3473 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3475 = stablehlo.add %3474, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3476 = stablehlo.multiply %3094, %3475 : tensor<144x9x128xbf16>
    %3477 = stablehlo.divide %3468, %3476 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3478 = stablehlo.multiply %arg27, %3089 : tensor<144x9x128xbf16>
    %3479 = stablehlo.add %3477, %3478 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3480 = stablehlo.multiply %3084, %3479 : tensor<144x9x128xbf16>
    %3481 = stablehlo.add %arg27, %3480 : tensor<144x9x128xbf16>
    %3482 = stablehlo.divide %2508, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3483 = stablehlo.select %3085, %2508, %3482 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3484 = stablehlo.multiply %3483, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3485 = stablehlo.multiply %arg95, %3091 : tensor<144x9x128xbf16>
    %3486 = stablehlo.add %3484, %3485 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3487 = stablehlo.multiply %3483, %3483 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3488 = stablehlo.multiply %3487, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3489 = stablehlo.multiply %arg161, %3098 : tensor<144x9x128xbf16>
    %3490 = stablehlo.add %3488, %3489 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3491 = stablehlo.divide %3490, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3492 = stablehlo.sqrt %3491 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3493 = stablehlo.add %3492, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3494 = stablehlo.multiply %3094, %3493 : tensor<144x9x128xbf16>
    %3495 = stablehlo.divide %3486, %3494 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3496 = stablehlo.multiply %arg28, %3089 : tensor<144x9x128xbf16>
    %3497 = stablehlo.add %3495, %3496 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3498 = stablehlo.multiply %3084, %3497 : tensor<144x9x128xbf16>
    %3499 = stablehlo.add %arg28, %3498 : tensor<144x9x128xbf16>
    %3500 = stablehlo.divide %2521, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3501 = stablehlo.select %2975, %2521, %3500 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %3502 = stablehlo.multiply %3501, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3503 = stablehlo.multiply %arg96, %2981 : tensor<144x2016xbf16>
    %3504 = stablehlo.add %3502, %3503 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3505 = stablehlo.multiply %3501, %3501 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3506 = stablehlo.multiply %3505, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3507 = stablehlo.multiply %arg162, %2988 : tensor<144x2016xbf16>
    %3508 = stablehlo.add %3506, %3507 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3509 = stablehlo.divide %3508, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3510 = stablehlo.sqrt %3509 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3511 = stablehlo.add %3510, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3512 = stablehlo.multiply %2984, %3511 : tensor<144x2016xbf16>
    %3513 = stablehlo.divide %3504, %3512 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3514 = stablehlo.multiply %arg29, %2979 : tensor<144x2016xbf16>
    %3515 = stablehlo.add %3513, %3514 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3516 = stablehlo.multiply %2974, %3515 : tensor<144x2016xbf16>
    %3517 = stablehlo.add %arg29, %3516 : tensor<144x2016xbf16>
    %3518 = stablehlo.divide %2530, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3519 = stablehlo.select %2975, %2530, %3518 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %3520 = stablehlo.multiply %3519, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3521 = stablehlo.multiply %arg97, %2981 : tensor<144x2016xbf16>
    %3522 = stablehlo.add %3520, %3521 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3523 = stablehlo.multiply %3519, %3519 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3524 = stablehlo.multiply %3523, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3525 = stablehlo.multiply %arg163, %2988 : tensor<144x2016xbf16>
    %3526 = stablehlo.add %3524, %3525 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3527 = stablehlo.divide %3526, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3528 = stablehlo.sqrt %3527 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3529 = stablehlo.add %3528, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3530 = stablehlo.multiply %2984, %3529 : tensor<144x2016xbf16>
    %3531 = stablehlo.divide %3522, %3530 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3532 = stablehlo.multiply %arg30, %2979 : tensor<144x2016xbf16>
    %3533 = stablehlo.add %3531, %3532 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3534 = stablehlo.multiply %2974, %3533 : tensor<144x2016xbf16>
    %3535 = stablehlo.add %arg30, %3534 : tensor<144x2016xbf16>
    %3536 = stablehlo.divide %2541, %3022 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3537 = stablehlo.select %3021, %2541, %3536 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xi1>, tensor<2016x144xbf16>
    %3538 = stablehlo.multiply %3537, %3025 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3539 = stablehlo.multiply %arg98, %3027 : tensor<2016x144xbf16>
    %3540 = stablehlo.add %3538, %3539 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3541 = stablehlo.multiply %3537, %3537 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3542 = stablehlo.multiply %3541, %3032 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3543 = stablehlo.multiply %arg164, %3034 : tensor<2016x144xbf16>
    %3544 = stablehlo.add %3542, %3543 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3545 = stablehlo.divide %3544, %3037 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3546 = stablehlo.sqrt %3545 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3547 = stablehlo.add %3546, %3040 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3548 = stablehlo.multiply %3030, %3547 : tensor<2016x144xbf16>
    %3549 = stablehlo.divide %3540, %3548 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3550 = stablehlo.multiply %arg31, %3025 : tensor<2016x144xbf16>
    %3551 = stablehlo.add %3549, %3550 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3552 = stablehlo.multiply %3020, %3551 : tensor<2016x144xbf16>
    %3553 = stablehlo.add %arg31, %3552 : tensor<2016x144xbf16>
    %3554 = stablehlo.divide %2550, %2938 : tensor<576xbf16>
    %3555 = stablehlo.select %2937, %2550, %3554 : tensor<576xi1>, tensor<576xbf16>
    %3556 = stablehlo.multiply %3555, %2941 : tensor<576xbf16>
    %3557 = stablehlo.multiply %arg99, %2943 : tensor<576xbf16>
    %3558 = stablehlo.add %3556, %3557 : tensor<576xbf16>
    %3559 = stablehlo.multiply %3555, %3555 : tensor<576xbf16>
    %3560 = stablehlo.multiply %3559, %2955 : tensor<576xbf16>
    %3561 = stablehlo.multiply %arg165, %2957 : tensor<576xbf16>
    %3562 = stablehlo.add %3560, %3561 : tensor<576xbf16>
    %3563 = stablehlo.divide %3562, %2963 : tensor<576xbf16>
    %3564 = stablehlo.sqrt %3563 : tensor<576xbf16>
    %3565 = stablehlo.add %3564, %2966 : tensor<576xbf16>
    %3566 = stablehlo.multiply %2953, %3565 : tensor<576xbf16>
    %3567 = stablehlo.divide %3558, %3566 : tensor<576xbf16>
    %3568 = stablehlo.multiply %arg32, %2941 : tensor<576xbf16>
    %3569 = stablehlo.add %3567, %3568 : tensor<576xbf16>
    %3570 = stablehlo.multiply %22, %3569 : tensor<576xbf16>
    %3571 = stablehlo.add %arg32, %3570 : tensor<576xbf16>
    %3572 = stablehlo.divide %2560, %2938 : tensor<576xbf16>
    %3573 = stablehlo.select %2937, %2560, %3572 : tensor<576xi1>, tensor<576xbf16>
    %3574 = stablehlo.multiply %3573, %2941 : tensor<576xbf16>
    %3575 = stablehlo.multiply %arg100, %2943 : tensor<576xbf16>
    %3576 = stablehlo.add %3574, %3575 : tensor<576xbf16>
    %3577 = stablehlo.multiply %3573, %3573 : tensor<576xbf16>
    %3578 = stablehlo.multiply %3577, %2955 : tensor<576xbf16>
    %3579 = stablehlo.multiply %arg166, %2957 : tensor<576xbf16>
    %3580 = stablehlo.add %3578, %3579 : tensor<576xbf16>
    %3581 = stablehlo.divide %3580, %2963 : tensor<576xbf16>
    %3582 = stablehlo.sqrt %3581 : tensor<576xbf16>
    %3583 = stablehlo.add %3582, %2966 : tensor<576xbf16>
    %3584 = stablehlo.multiply %2953, %3583 : tensor<576xbf16>
    %3585 = stablehlo.divide %3576, %3584 : tensor<576xbf16>
    %3586 = stablehlo.multiply %arg33, %2941 : tensor<576xbf16>
    %3587 = stablehlo.add %3585, %3586 : tensor<576xbf16>
    %3588 = stablehlo.multiply %22, %3587 : tensor<576xbf16>
    %3589 = stablehlo.add %arg33, %3588 : tensor<576xbf16>
    %3590 = stablehlo.divide %2572, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3591 = stablehlo.select %3085, %2572, %3590 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3592 = stablehlo.multiply %3591, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3593 = stablehlo.multiply %arg101, %3091 : tensor<144x9x128xbf16>
    %3594 = stablehlo.add %3592, %3593 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3595 = stablehlo.multiply %3591, %3591 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3596 = stablehlo.multiply %3595, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3597 = stablehlo.multiply %arg167, %3098 : tensor<144x9x128xbf16>
    %3598 = stablehlo.add %3596, %3597 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3599 = stablehlo.divide %3598, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3600 = stablehlo.sqrt %3599 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3601 = stablehlo.add %3600, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3602 = stablehlo.multiply %3094, %3601 : tensor<144x9x128xbf16>
    %3603 = stablehlo.divide %3594, %3602 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3604 = stablehlo.multiply %arg34, %3089 : tensor<144x9x128xbf16>
    %3605 = stablehlo.add %3603, %3604 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3606 = stablehlo.multiply %3084, %3605 : tensor<144x9x128xbf16>
    %3607 = stablehlo.add %arg34, %3606 : tensor<144x9x128xbf16>
    %3608 = stablehlo.divide %2584, %3114 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3609 = stablehlo.select %3113, %2584, %3608 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xi1>, tensor<9x128x144xbf16>
    %3610 = stablehlo.multiply %3609, %3117 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3611 = stablehlo.multiply %arg102, %3119 : tensor<9x128x144xbf16>
    %3612 = stablehlo.add %3610, %3611 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3613 = stablehlo.multiply %3609, %3609 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3614 = stablehlo.multiply %3613, %3124 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3615 = stablehlo.multiply %arg168, %3126 : tensor<9x128x144xbf16>
    %3616 = stablehlo.add %3614, %3615 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3617 = stablehlo.divide %3616, %3129 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3618 = stablehlo.sqrt %3617 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3619 = stablehlo.add %3618, %3132 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3620 = stablehlo.multiply %3122, %3619 : tensor<9x128x144xbf16>
    %3621 = stablehlo.divide %3612, %3620 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3622 = stablehlo.multiply %arg35, %3117 : tensor<9x128x144xbf16>
    %3623 = stablehlo.add %3621, %3622 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3624 = stablehlo.multiply %3112, %3623 : tensor<9x128x144xbf16>
    %3625 = stablehlo.add %arg35, %3624 : tensor<9x128x144xbf16>
    %3626 = stablehlo.divide %2594, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3627 = stablehlo.select %3085, %2594, %3626 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3628 = stablehlo.multiply %3627, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3629 = stablehlo.multiply %arg103, %3091 : tensor<144x9x128xbf16>
    %3630 = stablehlo.add %3628, %3629 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3631 = stablehlo.multiply %3627, %3627 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3632 = stablehlo.multiply %3631, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3633 = stablehlo.multiply %arg169, %3098 : tensor<144x9x128xbf16>
    %3634 = stablehlo.add %3632, %3633 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3635 = stablehlo.divide %3634, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3636 = stablehlo.sqrt %3635 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3637 = stablehlo.add %3636, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3638 = stablehlo.multiply %3094, %3637 : tensor<144x9x128xbf16>
    %3639 = stablehlo.divide %3630, %3638 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3640 = stablehlo.multiply %arg36, %3089 : tensor<144x9x128xbf16>
    %3641 = stablehlo.add %3639, %3640 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3642 = stablehlo.multiply %3084, %3641 : tensor<144x9x128xbf16>
    %3643 = stablehlo.add %arg36, %3642 : tensor<144x9x128xbf16>
    %3644 = stablehlo.divide %2604, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3645 = stablehlo.select %3085, %2604, %3644 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3646 = stablehlo.multiply %3645, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3647 = stablehlo.multiply %arg104, %3091 : tensor<144x9x128xbf16>
    %3648 = stablehlo.add %3646, %3647 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3649 = stablehlo.multiply %3645, %3645 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3650 = stablehlo.multiply %3649, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3651 = stablehlo.multiply %arg170, %3098 : tensor<144x9x128xbf16>
    %3652 = stablehlo.add %3650, %3651 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3653 = stablehlo.divide %3652, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3654 = stablehlo.sqrt %3653 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3655 = stablehlo.add %3654, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3656 = stablehlo.multiply %3094, %3655 : tensor<144x9x128xbf16>
    %3657 = stablehlo.divide %3648, %3656 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3658 = stablehlo.multiply %arg37, %3089 : tensor<144x9x128xbf16>
    %3659 = stablehlo.add %3657, %3658 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3660 = stablehlo.multiply %3084, %3659 : tensor<144x9x128xbf16>
    %3661 = stablehlo.add %arg37, %3660 : tensor<144x9x128xbf16>
    %3662 = stablehlo.divide %2617, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3663 = stablehlo.select %2975, %2617, %3662 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %3664 = stablehlo.multiply %3663, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3665 = stablehlo.multiply %arg105, %2981 : tensor<144x2016xbf16>
    %3666 = stablehlo.add %3664, %3665 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3667 = stablehlo.multiply %3663, %3663 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3668 = stablehlo.multiply %3667, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3669 = stablehlo.multiply %arg171, %2988 : tensor<144x2016xbf16>
    %3670 = stablehlo.add %3668, %3669 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3671 = stablehlo.divide %3670, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3672 = stablehlo.sqrt %3671 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3673 = stablehlo.add %3672, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3674 = stablehlo.multiply %2984, %3673 : tensor<144x2016xbf16>
    %3675 = stablehlo.divide %3666, %3674 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3676 = stablehlo.multiply %arg38, %2979 : tensor<144x2016xbf16>
    %3677 = stablehlo.add %3675, %3676 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3678 = stablehlo.multiply %2974, %3677 : tensor<144x2016xbf16>
    %3679 = stablehlo.add %arg38, %3678 : tensor<144x2016xbf16>
    %3680 = stablehlo.divide %2626, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3681 = stablehlo.select %2975, %2626, %3680 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %3682 = stablehlo.multiply %3681, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3683 = stablehlo.multiply %arg106, %2981 : tensor<144x2016xbf16>
    %3684 = stablehlo.add %3682, %3683 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3685 = stablehlo.multiply %3681, %3681 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3686 = stablehlo.multiply %3685, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3687 = stablehlo.multiply %arg172, %2988 : tensor<144x2016xbf16>
    %3688 = stablehlo.add %3686, %3687 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3689 = stablehlo.divide %3688, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3690 = stablehlo.sqrt %3689 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3691 = stablehlo.add %3690, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3692 = stablehlo.multiply %2984, %3691 : tensor<144x2016xbf16>
    %3693 = stablehlo.divide %3684, %3692 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3694 = stablehlo.multiply %arg39, %2979 : tensor<144x2016xbf16>
    %3695 = stablehlo.add %3693, %3694 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3696 = stablehlo.multiply %2974, %3695 : tensor<144x2016xbf16>
    %3697 = stablehlo.add %arg39, %3696 : tensor<144x2016xbf16>
    %3698 = stablehlo.divide %2637, %3022 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3699 = stablehlo.select %3021, %2637, %3698 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xi1>, tensor<2016x144xbf16>
    %3700 = stablehlo.multiply %3699, %3025 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3701 = stablehlo.multiply %arg107, %3027 : tensor<2016x144xbf16>
    %3702 = stablehlo.add %3700, %3701 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3703 = stablehlo.multiply %3699, %3699 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3704 = stablehlo.multiply %3703, %3032 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3705 = stablehlo.multiply %arg173, %3034 : tensor<2016x144xbf16>
    %3706 = stablehlo.add %3704, %3705 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3707 = stablehlo.divide %3706, %3037 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3708 = stablehlo.sqrt %3707 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3709 = stablehlo.add %3708, %3040 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3710 = stablehlo.multiply %3030, %3709 : tensor<2016x144xbf16>
    %3711 = stablehlo.divide %3702, %3710 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3712 = stablehlo.multiply %arg40, %3025 : tensor<2016x144xbf16>
    %3713 = stablehlo.add %3711, %3712 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3714 = stablehlo.multiply %3020, %3713 : tensor<2016x144xbf16>
    %3715 = stablehlo.add %arg40, %3714 : tensor<2016x144xbf16>
    %3716 = stablehlo.divide %2646, %2938 : tensor<576xbf16>
    %3717 = stablehlo.select %2937, %2646, %3716 : tensor<576xi1>, tensor<576xbf16>
    %3718 = stablehlo.multiply %3717, %2941 : tensor<576xbf16>
    %3719 = stablehlo.multiply %arg108, %2943 : tensor<576xbf16>
    %3720 = stablehlo.add %3718, %3719 : tensor<576xbf16>
    %3721 = stablehlo.multiply %3717, %3717 : tensor<576xbf16>
    %3722 = stablehlo.multiply %3721, %2955 : tensor<576xbf16>
    %3723 = stablehlo.multiply %arg174, %2957 : tensor<576xbf16>
    %3724 = stablehlo.add %3722, %3723 : tensor<576xbf16>
    %3725 = stablehlo.divide %3724, %2963 : tensor<576xbf16>
    %3726 = stablehlo.sqrt %3725 : tensor<576xbf16>
    %3727 = stablehlo.add %3726, %2966 : tensor<576xbf16>
    %3728 = stablehlo.multiply %2953, %3727 : tensor<576xbf16>
    %3729 = stablehlo.divide %3720, %3728 : tensor<576xbf16>
    %3730 = stablehlo.multiply %arg41, %2941 : tensor<576xbf16>
    %3731 = stablehlo.add %3729, %3730 : tensor<576xbf16>
    %3732 = stablehlo.multiply %22, %3731 : tensor<576xbf16>
    %3733 = stablehlo.add %arg41, %3732 : tensor<576xbf16>
    %3734 = stablehlo.divide %2656, %2938 : tensor<576xbf16>
    %3735 = stablehlo.select %2937, %2656, %3734 : tensor<576xi1>, tensor<576xbf16>
    %3736 = stablehlo.multiply %3735, %2941 : tensor<576xbf16>
    %3737 = stablehlo.multiply %arg109, %2943 : tensor<576xbf16>
    %3738 = stablehlo.add %3736, %3737 : tensor<576xbf16>
    %3739 = stablehlo.multiply %3735, %3735 : tensor<576xbf16>
    %3740 = stablehlo.multiply %3739, %2955 : tensor<576xbf16>
    %3741 = stablehlo.multiply %arg175, %2957 : tensor<576xbf16>
    %3742 = stablehlo.add %3740, %3741 : tensor<576xbf16>
    %3743 = stablehlo.divide %3742, %2963 : tensor<576xbf16>
    %3744 = stablehlo.sqrt %3743 : tensor<576xbf16>
    %3745 = stablehlo.add %3744, %2966 : tensor<576xbf16>
    %3746 = stablehlo.multiply %2953, %3745 : tensor<576xbf16>
    %3747 = stablehlo.divide %3738, %3746 : tensor<576xbf16>
    %3748 = stablehlo.multiply %arg42, %2941 : tensor<576xbf16>
    %3749 = stablehlo.add %3747, %3748 : tensor<576xbf16>
    %3750 = stablehlo.multiply %22, %3749 : tensor<576xbf16>
    %3751 = stablehlo.add %arg42, %3750 : tensor<576xbf16>
    %3752 = stablehlo.divide %2668, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3753 = stablehlo.select %3085, %2668, %3752 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3754 = stablehlo.multiply %3753, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3755 = stablehlo.multiply %arg110, %3091 : tensor<144x9x128xbf16>
    %3756 = stablehlo.add %3754, %3755 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3757 = stablehlo.multiply %3753, %3753 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3758 = stablehlo.multiply %3757, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3759 = stablehlo.multiply %arg176, %3098 : tensor<144x9x128xbf16>
    %3760 = stablehlo.add %3758, %3759 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3761 = stablehlo.divide %3760, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3762 = stablehlo.sqrt %3761 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3763 = stablehlo.add %3762, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3764 = stablehlo.multiply %3094, %3763 : tensor<144x9x128xbf16>
    %3765 = stablehlo.divide %3756, %3764 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3766 = stablehlo.multiply %arg43, %3089 : tensor<144x9x128xbf16>
    %3767 = stablehlo.add %3765, %3766 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3768 = stablehlo.multiply %3084, %3767 : tensor<144x9x128xbf16>
    %3769 = stablehlo.add %arg43, %3768 : tensor<144x9x128xbf16>
    %3770 = stablehlo.divide %2680, %3114 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3771 = stablehlo.select %3113, %2680, %3770 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xi1>, tensor<9x128x144xbf16>
    %3772 = stablehlo.multiply %3771, %3117 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3773 = stablehlo.multiply %arg111, %3119 : tensor<9x128x144xbf16>
    %3774 = stablehlo.add %3772, %3773 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3775 = stablehlo.multiply %3771, %3771 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3776 = stablehlo.multiply %3775, %3124 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3777 = stablehlo.multiply %arg177, %3126 : tensor<9x128x144xbf16>
    %3778 = stablehlo.add %3776, %3777 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3779 = stablehlo.divide %3778, %3129 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3780 = stablehlo.sqrt %3779 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3781 = stablehlo.add %3780, %3132 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3782 = stablehlo.multiply %3122, %3781 : tensor<9x128x144xbf16>
    %3783 = stablehlo.divide %3774, %3782 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3784 = stablehlo.multiply %arg44, %3117 : tensor<9x128x144xbf16>
    %3785 = stablehlo.add %3783, %3784 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3786 = stablehlo.multiply %3112, %3785 : tensor<9x128x144xbf16>
    %3787 = stablehlo.add %arg44, %3786 : tensor<9x128x144xbf16>
    %3788 = stablehlo.divide %2690, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3789 = stablehlo.select %3085, %2690, %3788 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3790 = stablehlo.multiply %3789, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3791 = stablehlo.multiply %arg112, %3091 : tensor<144x9x128xbf16>
    %3792 = stablehlo.add %3790, %3791 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3793 = stablehlo.multiply %3789, %3789 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3794 = stablehlo.multiply %3793, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3795 = stablehlo.multiply %arg178, %3098 : tensor<144x9x128xbf16>
    %3796 = stablehlo.add %3794, %3795 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3797 = stablehlo.divide %3796, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3798 = stablehlo.sqrt %3797 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3799 = stablehlo.add %3798, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3800 = stablehlo.multiply %3094, %3799 : tensor<144x9x128xbf16>
    %3801 = stablehlo.divide %3792, %3800 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3802 = stablehlo.multiply %arg45, %3089 : tensor<144x9x128xbf16>
    %3803 = stablehlo.add %3801, %3802 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3804 = stablehlo.multiply %3084, %3803 : tensor<144x9x128xbf16>
    %3805 = stablehlo.add %arg45, %3804 : tensor<144x9x128xbf16>
    %3806 = stablehlo.divide %2700, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3807 = stablehlo.select %3085, %2700, %3806 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3808 = stablehlo.multiply %3807, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3809 = stablehlo.multiply %arg113, %3091 : tensor<144x9x128xbf16>
    %3810 = stablehlo.add %3808, %3809 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3811 = stablehlo.multiply %3807, %3807 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3812 = stablehlo.multiply %3811, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3813 = stablehlo.multiply %arg179, %3098 : tensor<144x9x128xbf16>
    %3814 = stablehlo.add %3812, %3813 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3815 = stablehlo.divide %3814, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3816 = stablehlo.sqrt %3815 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3817 = stablehlo.add %3816, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3818 = stablehlo.multiply %3094, %3817 : tensor<144x9x128xbf16>
    %3819 = stablehlo.divide %3810, %3818 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3820 = stablehlo.multiply %arg46, %3089 : tensor<144x9x128xbf16>
    %3821 = stablehlo.add %3819, %3820 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3822 = stablehlo.multiply %3084, %3821 : tensor<144x9x128xbf16>
    %3823 = stablehlo.add %arg46, %3822 : tensor<144x9x128xbf16>
    %3824 = stablehlo.divide %2713, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3825 = stablehlo.select %2975, %2713, %3824 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %3826 = stablehlo.multiply %3825, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3827 = stablehlo.multiply %arg114, %2981 : tensor<144x2016xbf16>
    %3828 = stablehlo.add %3826, %3827 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3829 = stablehlo.multiply %3825, %3825 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3830 = stablehlo.multiply %3829, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3831 = stablehlo.multiply %arg180, %2988 : tensor<144x2016xbf16>
    %3832 = stablehlo.add %3830, %3831 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3833 = stablehlo.divide %3832, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3834 = stablehlo.sqrt %3833 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3835 = stablehlo.add %3834, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3836 = stablehlo.multiply %2984, %3835 : tensor<144x2016xbf16>
    %3837 = stablehlo.divide %3828, %3836 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3838 = stablehlo.multiply %arg47, %2979 : tensor<144x2016xbf16>
    %3839 = stablehlo.add %3837, %3838 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3840 = stablehlo.multiply %2974, %3839 : tensor<144x2016xbf16>
    %3841 = stablehlo.add %arg47, %3840 : tensor<144x2016xbf16>
    %3842 = stablehlo.divide %2722, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3843 = stablehlo.select %2975, %2722, %3842 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %3844 = stablehlo.multiply %3843, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3845 = stablehlo.multiply %arg115, %2981 : tensor<144x2016xbf16>
    %3846 = stablehlo.add %3844, %3845 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3847 = stablehlo.multiply %3843, %3843 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3848 = stablehlo.multiply %3847, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3849 = stablehlo.multiply %arg181, %2988 : tensor<144x2016xbf16>
    %3850 = stablehlo.add %3848, %3849 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3851 = stablehlo.divide %3850, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3852 = stablehlo.sqrt %3851 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3853 = stablehlo.add %3852, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3854 = stablehlo.multiply %2984, %3853 : tensor<144x2016xbf16>
    %3855 = stablehlo.divide %3846, %3854 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3856 = stablehlo.multiply %arg48, %2979 : tensor<144x2016xbf16>
    %3857 = stablehlo.add %3855, %3856 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3858 = stablehlo.multiply %2974, %3857 : tensor<144x2016xbf16>
    %3859 = stablehlo.add %arg48, %3858 : tensor<144x2016xbf16>
    %3860 = stablehlo.divide %2733, %3022 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3861 = stablehlo.select %3021, %2733, %3860 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xi1>, tensor<2016x144xbf16>
    %3862 = stablehlo.multiply %3861, %3025 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3863 = stablehlo.multiply %arg116, %3027 : tensor<2016x144xbf16>
    %3864 = stablehlo.add %3862, %3863 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3865 = stablehlo.multiply %3861, %3861 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3866 = stablehlo.multiply %3865, %3032 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3867 = stablehlo.multiply %arg182, %3034 : tensor<2016x144xbf16>
    %3868 = stablehlo.add %3866, %3867 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3869 = stablehlo.divide %3868, %3037 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3870 = stablehlo.sqrt %3869 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3871 = stablehlo.add %3870, %3040 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3872 = stablehlo.multiply %3030, %3871 : tensor<2016x144xbf16>
    %3873 = stablehlo.divide %3864, %3872 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3874 = stablehlo.multiply %arg49, %3025 : tensor<2016x144xbf16>
    %3875 = stablehlo.add %3873, %3874 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %3876 = stablehlo.multiply %3020, %3875 : tensor<2016x144xbf16>
    %3877 = stablehlo.add %arg49, %3876 : tensor<2016x144xbf16>
    %3878 = stablehlo.divide %2742, %2938 : tensor<576xbf16>
    %3879 = stablehlo.select %2937, %2742, %3878 : tensor<576xi1>, tensor<576xbf16>
    %3880 = stablehlo.multiply %3879, %2941 : tensor<576xbf16>
    %3881 = stablehlo.multiply %arg117, %2943 : tensor<576xbf16>
    %3882 = stablehlo.add %3880, %3881 : tensor<576xbf16>
    %3883 = stablehlo.multiply %3879, %3879 : tensor<576xbf16>
    %3884 = stablehlo.multiply %3883, %2955 : tensor<576xbf16>
    %3885 = stablehlo.multiply %arg183, %2957 : tensor<576xbf16>
    %3886 = stablehlo.add %3884, %3885 : tensor<576xbf16>
    %3887 = stablehlo.divide %3886, %2963 : tensor<576xbf16>
    %3888 = stablehlo.sqrt %3887 : tensor<576xbf16>
    %3889 = stablehlo.add %3888, %2966 : tensor<576xbf16>
    %3890 = stablehlo.multiply %2953, %3889 : tensor<576xbf16>
    %3891 = stablehlo.divide %3882, %3890 : tensor<576xbf16>
    %3892 = stablehlo.multiply %arg50, %2941 : tensor<576xbf16>
    %3893 = stablehlo.add %3891, %3892 : tensor<576xbf16>
    %3894 = stablehlo.multiply %22, %3893 : tensor<576xbf16>
    %3895 = stablehlo.add %arg50, %3894 : tensor<576xbf16>
    %3896 = stablehlo.divide %2752, %2938 : tensor<576xbf16>
    %3897 = stablehlo.select %2937, %2752, %3896 : tensor<576xi1>, tensor<576xbf16>
    %3898 = stablehlo.multiply %3897, %2941 : tensor<576xbf16>
    %3899 = stablehlo.multiply %arg118, %2943 : tensor<576xbf16>
    %3900 = stablehlo.add %3898, %3899 : tensor<576xbf16>
    %3901 = stablehlo.multiply %3897, %3897 : tensor<576xbf16>
    %3902 = stablehlo.multiply %3901, %2955 : tensor<576xbf16>
    %3903 = stablehlo.multiply %arg184, %2957 : tensor<576xbf16>
    %3904 = stablehlo.add %3902, %3903 : tensor<576xbf16>
    %3905 = stablehlo.divide %3904, %2963 : tensor<576xbf16>
    %3906 = stablehlo.sqrt %3905 : tensor<576xbf16>
    %3907 = stablehlo.add %3906, %2966 : tensor<576xbf16>
    %3908 = stablehlo.multiply %2953, %3907 : tensor<576xbf16>
    %3909 = stablehlo.divide %3900, %3908 : tensor<576xbf16>
    %3910 = stablehlo.multiply %arg51, %2941 : tensor<576xbf16>
    %3911 = stablehlo.add %3909, %3910 : tensor<576xbf16>
    %3912 = stablehlo.multiply %22, %3911 : tensor<576xbf16>
    %3913 = stablehlo.add %arg51, %3912 : tensor<576xbf16>
    %3914 = stablehlo.divide %2764, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3915 = stablehlo.select %3085, %2764, %3914 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3916 = stablehlo.multiply %3915, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3917 = stablehlo.multiply %arg119, %3091 : tensor<144x9x128xbf16>
    %3918 = stablehlo.add %3916, %3917 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3919 = stablehlo.multiply %3915, %3915 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3920 = stablehlo.multiply %3919, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3921 = stablehlo.multiply %arg185, %3098 : tensor<144x9x128xbf16>
    %3922 = stablehlo.add %3920, %3921 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3923 = stablehlo.divide %3922, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3924 = stablehlo.sqrt %3923 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3925 = stablehlo.add %3924, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3926 = stablehlo.multiply %3094, %3925 : tensor<144x9x128xbf16>
    %3927 = stablehlo.divide %3918, %3926 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3928 = stablehlo.multiply %arg52, %3089 : tensor<144x9x128xbf16>
    %3929 = stablehlo.add %3927, %3928 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3930 = stablehlo.multiply %3084, %3929 : tensor<144x9x128xbf16>
    %3931 = stablehlo.add %arg52, %3930 : tensor<144x9x128xbf16>
    %3932 = stablehlo.divide %2776, %3114 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3933 = stablehlo.select %3113, %2776, %3932 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xi1>, tensor<9x128x144xbf16>
    %3934 = stablehlo.multiply %3933, %3117 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3935 = stablehlo.multiply %arg120, %3119 : tensor<9x128x144xbf16>
    %3936 = stablehlo.add %3934, %3935 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3937 = stablehlo.multiply %3933, %3933 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3938 = stablehlo.multiply %3937, %3124 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3939 = stablehlo.multiply %arg186, %3126 : tensor<9x128x144xbf16>
    %3940 = stablehlo.add %3938, %3939 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3941 = stablehlo.divide %3940, %3129 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3942 = stablehlo.sqrt %3941 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3943 = stablehlo.add %3942, %3132 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3944 = stablehlo.multiply %3122, %3943 : tensor<9x128x144xbf16>
    %3945 = stablehlo.divide %3936, %3944 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3946 = stablehlo.multiply %arg53, %3117 : tensor<9x128x144xbf16>
    %3947 = stablehlo.add %3945, %3946 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %3948 = stablehlo.multiply %3112, %3947 : tensor<9x128x144xbf16>
    %3949 = stablehlo.add %arg53, %3948 : tensor<9x128x144xbf16>
    %3950 = stablehlo.divide %2786, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3951 = stablehlo.select %3085, %2786, %3950 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3952 = stablehlo.multiply %3951, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3953 = stablehlo.multiply %arg121, %3091 : tensor<144x9x128xbf16>
    %3954 = stablehlo.add %3952, %3953 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3955 = stablehlo.multiply %3951, %3951 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3956 = stablehlo.multiply %3955, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3957 = stablehlo.multiply %arg187, %3098 : tensor<144x9x128xbf16>
    %3958 = stablehlo.add %3956, %3957 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3959 = stablehlo.divide %3958, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3960 = stablehlo.sqrt %3959 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3961 = stablehlo.add %3960, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3962 = stablehlo.multiply %3094, %3961 : tensor<144x9x128xbf16>
    %3963 = stablehlo.divide %3954, %3962 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3964 = stablehlo.multiply %arg54, %3089 : tensor<144x9x128xbf16>
    %3965 = stablehlo.add %3963, %3964 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3966 = stablehlo.multiply %3084, %3965 : tensor<144x9x128xbf16>
    %3967 = stablehlo.add %arg54, %3966 : tensor<144x9x128xbf16>
    %3968 = stablehlo.divide %2796, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3969 = stablehlo.select %3085, %2796, %3968 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %3970 = stablehlo.multiply %3969, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3971 = stablehlo.multiply %arg122, %3091 : tensor<144x9x128xbf16>
    %3972 = stablehlo.add %3970, %3971 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3973 = stablehlo.multiply %3969, %3969 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3974 = stablehlo.multiply %3973, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3975 = stablehlo.multiply %arg188, %3098 : tensor<144x9x128xbf16>
    %3976 = stablehlo.add %3974, %3975 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3977 = stablehlo.divide %3976, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3978 = stablehlo.sqrt %3977 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3979 = stablehlo.add %3978, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3980 = stablehlo.multiply %3094, %3979 : tensor<144x9x128xbf16>
    %3981 = stablehlo.divide %3972, %3980 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3982 = stablehlo.multiply %arg55, %3089 : tensor<144x9x128xbf16>
    %3983 = stablehlo.add %3981, %3982 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %3984 = stablehlo.multiply %3084, %3983 : tensor<144x9x128xbf16>
    %3985 = stablehlo.add %arg55, %3984 : tensor<144x9x128xbf16>
    %3986 = stablehlo.divide %2809, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3987 = stablehlo.select %2975, %2809, %3986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %3988 = stablehlo.multiply %3987, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3989 = stablehlo.multiply %arg123, %2981 : tensor<144x2016xbf16>
    %3990 = stablehlo.add %3988, %3989 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3991 = stablehlo.multiply %3987, %3987 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3992 = stablehlo.multiply %3991, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3993 = stablehlo.multiply %arg189, %2988 : tensor<144x2016xbf16>
    %3994 = stablehlo.add %3992, %3993 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3995 = stablehlo.divide %3994, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3996 = stablehlo.sqrt %3995 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3997 = stablehlo.add %3996, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %3998 = stablehlo.multiply %2984, %3997 : tensor<144x2016xbf16>
    %3999 = stablehlo.divide %3990, %3998 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %4000 = stablehlo.multiply %arg56, %2979 : tensor<144x2016xbf16>
    %4001 = stablehlo.add %3999, %4000 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %4002 = stablehlo.multiply %2974, %4001 : tensor<144x2016xbf16>
    %4003 = stablehlo.add %arg56, %4002 : tensor<144x2016xbf16>
    %4004 = stablehlo.divide %2818, %2976 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %4005 = stablehlo.select %2975, %2818, %4004 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xi1>, tensor<144x2016xbf16>
    %4006 = stablehlo.multiply %4005, %2979 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %4007 = stablehlo.multiply %arg124, %2981 : tensor<144x2016xbf16>
    %4008 = stablehlo.add %4006, %4007 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %4009 = stablehlo.multiply %4005, %4005 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %4010 = stablehlo.multiply %4009, %2986 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %4011 = stablehlo.multiply %arg190, %2988 : tensor<144x2016xbf16>
    %4012 = stablehlo.add %4010, %4011 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %4013 = stablehlo.divide %4012, %2991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %4014 = stablehlo.sqrt %4013 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %4015 = stablehlo.add %4014, %2994 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %4016 = stablehlo.multiply %2984, %4015 : tensor<144x2016xbf16>
    %4017 = stablehlo.divide %4008, %4016 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %4018 = stablehlo.multiply %arg57, %2979 : tensor<144x2016xbf16>
    %4019 = stablehlo.add %4017, %4018 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,2016]{0,1}"} : tensor<144x2016xbf16>
    %4020 = stablehlo.multiply %2974, %4019 : tensor<144x2016xbf16>
    %4021 = stablehlo.add %arg57, %4020 : tensor<144x2016xbf16>
    %4022 = stablehlo.divide %2829, %3022 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %4023 = stablehlo.select %3021, %2829, %4022 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xi1>, tensor<2016x144xbf16>
    %4024 = stablehlo.multiply %4023, %3025 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %4025 = stablehlo.multiply %arg125, %3027 : tensor<2016x144xbf16>
    %4026 = stablehlo.add %4024, %4025 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %4027 = stablehlo.multiply %4023, %4023 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %4028 = stablehlo.multiply %4027, %3032 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %4029 = stablehlo.multiply %arg191, %3034 : tensor<2016x144xbf16>
    %4030 = stablehlo.add %4028, %4029 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %4031 = stablehlo.divide %4030, %3037 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %4032 = stablehlo.sqrt %4031 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %4033 = stablehlo.add %4032, %3040 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %4034 = stablehlo.multiply %3030, %4033 : tensor<2016x144xbf16>
    %4035 = stablehlo.divide %4026, %4034 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %4036 = stablehlo.multiply %arg58, %3025 : tensor<2016x144xbf16>
    %4037 = stablehlo.add %4035, %4036 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2016,144]{0,1}"} : tensor<2016x144xbf16>
    %4038 = stablehlo.multiply %3020, %4037 : tensor<2016x144xbf16>
    %4039 = stablehlo.add %arg58, %4038 : tensor<2016x144xbf16>
    %4040 = stablehlo.divide %2838, %2938 : tensor<576xbf16>
    %4041 = stablehlo.select %2937, %2838, %4040 : tensor<576xi1>, tensor<576xbf16>
    %4042 = stablehlo.multiply %4041, %2941 : tensor<576xbf16>
    %4043 = stablehlo.multiply %arg126, %2943 : tensor<576xbf16>
    %4044 = stablehlo.add %4042, %4043 : tensor<576xbf16>
    %4045 = stablehlo.multiply %4041, %4041 : tensor<576xbf16>
    %4046 = stablehlo.multiply %4045, %2955 : tensor<576xbf16>
    %4047 = stablehlo.multiply %arg192, %2957 : tensor<576xbf16>
    %4048 = stablehlo.add %4046, %4047 : tensor<576xbf16>
    %4049 = stablehlo.divide %4048, %2963 : tensor<576xbf16>
    %4050 = stablehlo.sqrt %4049 : tensor<576xbf16>
    %4051 = stablehlo.add %4050, %2966 : tensor<576xbf16>
    %4052 = stablehlo.multiply %2953, %4051 : tensor<576xbf16>
    %4053 = stablehlo.divide %4044, %4052 : tensor<576xbf16>
    %4054 = stablehlo.multiply %arg59, %2941 : tensor<576xbf16>
    %4055 = stablehlo.add %4053, %4054 : tensor<576xbf16>
    %4056 = stablehlo.multiply %22, %4055 : tensor<576xbf16>
    %4057 = stablehlo.add %arg59, %4056 : tensor<576xbf16>
    %4058 = stablehlo.divide %2848, %2938 : tensor<576xbf16>
    %4059 = stablehlo.select %2937, %2848, %4058 : tensor<576xi1>, tensor<576xbf16>
    %4060 = stablehlo.multiply %4059, %2941 : tensor<576xbf16>
    %4061 = stablehlo.multiply %arg127, %2943 : tensor<576xbf16>
    %4062 = stablehlo.add %4060, %4061 : tensor<576xbf16>
    %4063 = stablehlo.multiply %4059, %4059 : tensor<576xbf16>
    %4064 = stablehlo.multiply %4063, %2955 : tensor<576xbf16>
    %4065 = stablehlo.multiply %arg193, %2957 : tensor<576xbf16>
    %4066 = stablehlo.add %4064, %4065 : tensor<576xbf16>
    %4067 = stablehlo.divide %4066, %2963 : tensor<576xbf16>
    %4068 = stablehlo.sqrt %4067 : tensor<576xbf16>
    %4069 = stablehlo.add %4068, %2966 : tensor<576xbf16>
    %4070 = stablehlo.multiply %2953, %4069 : tensor<576xbf16>
    %4071 = stablehlo.divide %4062, %4070 : tensor<576xbf16>
    %4072 = stablehlo.multiply %arg60, %2941 : tensor<576xbf16>
    %4073 = stablehlo.add %4071, %4072 : tensor<576xbf16>
    %4074 = stablehlo.multiply %22, %4073 : tensor<576xbf16>
    %4075 = stablehlo.add %arg60, %4074 : tensor<576xbf16>
    %4076 = stablehlo.divide %2860, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4077 = stablehlo.select %3085, %2860, %4076 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %4078 = stablehlo.multiply %4077, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4079 = stablehlo.multiply %arg128, %3091 : tensor<144x9x128xbf16>
    %4080 = stablehlo.add %4078, %4079 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4081 = stablehlo.multiply %4077, %4077 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4082 = stablehlo.multiply %4081, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4083 = stablehlo.multiply %arg194, %3098 : tensor<144x9x128xbf16>
    %4084 = stablehlo.add %4082, %4083 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4085 = stablehlo.divide %4084, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4086 = stablehlo.sqrt %4085 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4087 = stablehlo.add %4086, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4088 = stablehlo.multiply %3094, %4087 : tensor<144x9x128xbf16>
    %4089 = stablehlo.divide %4080, %4088 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4090 = stablehlo.multiply %arg61, %3089 : tensor<144x9x128xbf16>
    %4091 = stablehlo.add %4089, %4090 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4092 = stablehlo.multiply %3084, %4091 : tensor<144x9x128xbf16>
    %4093 = stablehlo.add %arg61, %4092 : tensor<144x9x128xbf16>
    %4094 = stablehlo.divide %2872, %3114 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %4095 = stablehlo.select %3113, %2872, %4094 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xi1>, tensor<9x128x144xbf16>
    %4096 = stablehlo.multiply %4095, %3117 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %4097 = stablehlo.multiply %arg129, %3119 : tensor<9x128x144xbf16>
    %4098 = stablehlo.add %4096, %4097 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %4099 = stablehlo.multiply %4095, %4095 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %4100 = stablehlo.multiply %4099, %3124 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %4101 = stablehlo.multiply %arg195, %3126 : tensor<9x128x144xbf16>
    %4102 = stablehlo.add %4100, %4101 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %4103 = stablehlo.divide %4102, %3129 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %4104 = stablehlo.sqrt %4103 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %4105 = stablehlo.add %4104, %3132 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %4106 = stablehlo.multiply %3122, %4105 : tensor<9x128x144xbf16>
    %4107 = stablehlo.divide %4098, %4106 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %4108 = stablehlo.multiply %arg62, %3117 : tensor<9x128x144xbf16>
    %4109 = stablehlo.add %4107, %4108 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "bf16[9,128,144]{1,0,2}"} : tensor<9x128x144xbf16>
    %4110 = stablehlo.multiply %3112, %4109 : tensor<9x128x144xbf16>
    %4111 = stablehlo.add %arg62, %4110 : tensor<9x128x144xbf16>
    %4112 = stablehlo.divide %2882, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4113 = stablehlo.select %3085, %2882, %4112 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %4114 = stablehlo.multiply %4113, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4115 = stablehlo.multiply %arg130, %3091 : tensor<144x9x128xbf16>
    %4116 = stablehlo.add %4114, %4115 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4117 = stablehlo.multiply %4113, %4113 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4118 = stablehlo.multiply %4117, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4119 = stablehlo.multiply %arg196, %3098 : tensor<144x9x128xbf16>
    %4120 = stablehlo.add %4118, %4119 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4121 = stablehlo.divide %4120, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4122 = stablehlo.sqrt %4121 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4123 = stablehlo.add %4122, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4124 = stablehlo.multiply %3094, %4123 : tensor<144x9x128xbf16>
    %4125 = stablehlo.divide %4116, %4124 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4126 = stablehlo.multiply %arg63, %3089 : tensor<144x9x128xbf16>
    %4127 = stablehlo.add %4125, %4126 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4128 = stablehlo.multiply %3084, %4127 : tensor<144x9x128xbf16>
    %4129 = stablehlo.add %arg63, %4128 : tensor<144x9x128xbf16>
    %4130 = stablehlo.divide %2892, %3086 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4131 = stablehlo.select %3085, %2892, %4130 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xi1>, tensor<144x9x128xbf16>
    %4132 = stablehlo.multiply %4131, %3089 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4133 = stablehlo.multiply %arg131, %3091 : tensor<144x9x128xbf16>
    %4134 = stablehlo.add %4132, %4133 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4135 = stablehlo.multiply %4131, %4131 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4136 = stablehlo.multiply %4135, %3096 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4137 = stablehlo.multiply %arg197, %3098 : tensor<144x9x128xbf16>
    %4138 = stablehlo.add %4136, %4137 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4139 = stablehlo.divide %4138, %3101 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4140 = stablehlo.sqrt %4139 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4141 = stablehlo.add %4140, %3104 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4142 = stablehlo.multiply %3094, %4141 : tensor<144x9x128xbf16>
    %4143 = stablehlo.divide %4134, %4142 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4144 = stablehlo.multiply %arg64, %3089 : tensor<144x9x128xbf16>
    %4145 = stablehlo.add %4143, %4144 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[144,9,128]{0,2,1}"} : tensor<144x9x128xbf16>
    %4146 = stablehlo.multiply %3084, %4145 : tensor<144x9x128xbf16>
    %4147 = stablehlo.add %arg64, %4146 : tensor<144x9x128xbf16>
    %4148 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<144x128256xbf16>
    %4149 = stablehlo.broadcast_in_dim %2936, dims = [] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "pred[144,128256]{0,1}"} : (tensor<i1>) -> tensor<144x128256xi1>
    %4150 = stablehlo.broadcast_in_dim %2935, dims = [] : (tensor<bf16>) -> tensor<144x128256xbf16>
    %4151 = stablehlo.divide %2901, %4150 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,128256]{0,1}"} : tensor<144x128256xbf16>
    %4152 = stablehlo.select %4149, %2901, %4151 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,128256]{0,1}"} : tensor<144x128256xi1>, tensor<144x128256xbf16>
    %4153 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<bf16>) -> tensor<144x128256xbf16>
    %4154 = stablehlo.multiply %4152, %4153 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,128256]{0,1}"} : tensor<144x128256xbf16>
    %4155 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<bf16>) -> tensor<144x128256xbf16>
    %4156 = stablehlo.multiply %arg132, %4155 : tensor<144x128256xbf16>
    %4157 = stablehlo.add %4154, %4156 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,128256]{0,1}"} : tensor<144x128256xbf16>
    %4158 = stablehlo.broadcast_in_dim %2952, dims = [] : (tensor<bf16>) -> tensor<144x128256xbf16>
    %4159 = stablehlo.multiply %4152, %4152 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,128256]{0,1}"} : tensor<144x128256xbf16>
    %4160 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<bf16>) -> tensor<144x128256xbf16>
    %4161 = stablehlo.multiply %4159, %4160 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,128256]{0,1}"} : tensor<144x128256xbf16>
    %4162 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<bf16>) -> tensor<144x128256xbf16>
    %4163 = stablehlo.multiply %arg198, %4162 : tensor<144x128256xbf16>
    %4164 = stablehlo.add %4161, %4163 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,128256]{0,1}"} : tensor<144x128256xbf16>
    %4165 = stablehlo.broadcast_in_dim %2962, dims = [] : (tensor<bf16>) -> tensor<144x128256xbf16>
    %4166 = stablehlo.divide %4164, %4165 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,128256]{0,1}"} : tensor<144x128256xbf16>
    %4167 = stablehlo.sqrt %4166 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,128256]{0,1}"} : tensor<144x128256xbf16>
    %4168 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<bf16>) -> tensor<144x128256xbf16>
    %4169 = stablehlo.add %4167, %4168 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,128256]{0,1}"} : tensor<144x128256xbf16>
    %4170 = stablehlo.multiply %4158, %4169 : tensor<144x128256xbf16>
    %4171 = stablehlo.divide %4157, %4170 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,128256]{0,1}"} : tensor<144x128256xbf16>
    %4172 = stablehlo.multiply %arg65, %4153 : tensor<144x128256xbf16>
    %4173 = stablehlo.add %4171, %4172 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[144,128256]{0,1}"} : tensor<144x128256xbf16>
    %4174 = stablehlo.multiply %4148, %4173 : tensor<144x128256xbf16>
    %4175 = stablehlo.add %arg65, %4174 : tensor<144x128256xbf16>
    %4176 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<bf16>) -> tensor<128256x144xbf16>
    %4177 = stablehlo.broadcast_in_dim %2936, dims = [] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "pred[128256,144]{0,1}"} : (tensor<i1>) -> tensor<128256x144xi1>
    %4178 = stablehlo.broadcast_in_dim %2935, dims = [] : (tensor<bf16>) -> tensor<128256x144xbf16>
    %4179 = stablehlo.divide %2928, %4178 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128256,144]{0,1}"} : tensor<128256x144xbf16>
    %4180 = stablehlo.select %4177, %2928, %4179 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128256,144]{0,1}"} : tensor<128256x144xi1>, tensor<128256x144xbf16>
    %4181 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<bf16>) -> tensor<128256x144xbf16>
    %4182 = stablehlo.multiply %4180, %4181 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128256,144]{0,1}"} : tensor<128256x144xbf16>
    %4183 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<bf16>) -> tensor<128256x144xbf16>
    %4184 = stablehlo.multiply %arg133, %4183 : tensor<128256x144xbf16>
    %4185 = stablehlo.add %4182, %4184 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128256,144]{0,1}"} : tensor<128256x144xbf16>
    %4186 = stablehlo.broadcast_in_dim %2952, dims = [] : (tensor<bf16>) -> tensor<128256x144xbf16>
    %4187 = stablehlo.multiply %4180, %4180 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128256,144]{0,1}"} : tensor<128256x144xbf16>
    %4188 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<bf16>) -> tensor<128256x144xbf16>
    %4189 = stablehlo.multiply %4187, %4188 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128256,144]{0,1}"} : tensor<128256x144xbf16>
    %4190 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<bf16>) -> tensor<128256x144xbf16>
    %4191 = stablehlo.multiply %arg199, %4190 : tensor<128256x144xbf16>
    %4192 = stablehlo.add %4189, %4191 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128256,144]{0,1}"} : tensor<128256x144xbf16>
    %4193 = stablehlo.broadcast_in_dim %2962, dims = [] : (tensor<bf16>) -> tensor<128256x144xbf16>
    %4194 = stablehlo.divide %4192, %4193 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128256,144]{0,1}"} : tensor<128256x144xbf16>
    %4195 = stablehlo.sqrt %4194 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128256,144]{0,1}"} : tensor<128256x144xbf16>
    %4196 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<bf16>) -> tensor<128256x144xbf16>
    %4197 = stablehlo.add %4195, %4196 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128256,144]{0,1}"} : tensor<128256x144xbf16>
    %4198 = stablehlo.multiply %4186, %4197 : tensor<128256x144xbf16>
    %4199 = stablehlo.divide %4185, %4198 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128256,144]{0,1}"} : tensor<128256x144xbf16>
    %4200 = stablehlo.multiply %arg66, %4181 : tensor<128256x144xbf16>
    %4201 = stablehlo.add %4199, %4200 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128256,144]{0,1}"} : tensor<128256x144xbf16>
    %4202 = stablehlo.multiply %4176, %4201 : tensor<128256x144xbf16>
    %4203 = stablehlo.add %arg66, %4202 : tensor<128256x144xbf16>
    %4204 = stablehlo.compare  LT, %arg200, %c_31 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %4205 = stablehlo.add %arg200, %c : tensor<i32>
    %4206 = stablehlo.select %4204, %4205, %c_31 : tensor<i1>, tensor<i32>
    %4207 = stablehlo.convert %2954 : (tensor<576xbf16>) -> tensor<576xf32>
    %4208 = stablehlo.reduce(%4207 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4209 = stablehlo.convert %4208 : (tensor<f32>) -> tensor<bf16>
    %4210 = stablehlo.convert %2985 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4211 = stablehlo.reduce(%4210 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4212 = "stablehlo.all_reduce"(%4211) <{channel_handle = #stablehlo.channel_handle<handle = 219, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4213 = stablehlo.convert %4212 : (tensor<f32>) -> tensor<bf16>
    %4214 = stablehlo.add %4209, %4213 : tensor<bf16>
    %4215 = stablehlo.convert %3007 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4216 = stablehlo.reduce(%4215 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4217 = "stablehlo.all_reduce"(%4216) <{channel_handle = #stablehlo.channel_handle<handle = 220, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4218 = stablehlo.convert %4217 : (tensor<f32>) -> tensor<bf16>
    %4219 = stablehlo.add %4214, %4218 : tensor<bf16>
    %4220 = stablehlo.convert %3031 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4221 = stablehlo.reduce(%4220 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4222 = "stablehlo.all_reduce"(%4221) <{channel_handle = #stablehlo.channel_handle<handle = 221, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4223 = stablehlo.convert %4222 : (tensor<f32>) -> tensor<bf16>
    %4224 = stablehlo.add %4219, %4223 : tensor<bf16>
    %4225 = stablehlo.convert %3053 : (tensor<576xbf16>) -> tensor<576xf32>
    %4226 = stablehlo.reduce(%4225 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4227 = stablehlo.convert %4226 : (tensor<f32>) -> tensor<bf16>
    %4228 = stablehlo.add %4224, %4227 : tensor<bf16>
    %4229 = stablehlo.convert %3071 : (tensor<576xbf16>) -> tensor<576xf32>
    %4230 = stablehlo.reduce(%4229 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4231 = stablehlo.convert %4230 : (tensor<f32>) -> tensor<bf16>
    %4232 = stablehlo.add %4228, %4231 : tensor<bf16>
    %4233 = stablehlo.convert %3095 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4234 = stablehlo.reduce(%4233 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4235 = "stablehlo.all_reduce"(%4234) <{channel_handle = #stablehlo.channel_handle<handle = 222, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4236 = stablehlo.convert %4235 : (tensor<f32>) -> tensor<bf16>
    %4237 = stablehlo.add %4232, %4236 : tensor<bf16>
    %4238 = stablehlo.convert %3123 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4239 = stablehlo.reduce(%4238 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4240 = "stablehlo.all_reduce"(%4239) <{channel_handle = #stablehlo.channel_handle<handle = 223, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4241 = stablehlo.convert %4240 : (tensor<f32>) -> tensor<bf16>
    %4242 = stablehlo.add %4237, %4241 : tensor<bf16>
    %4243 = stablehlo.convert %3145 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4244 = stablehlo.reduce(%4243 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4245 = "stablehlo.all_reduce"(%4244) <{channel_handle = #stablehlo.channel_handle<handle = 224, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4246 = stablehlo.convert %4245 : (tensor<f32>) -> tensor<bf16>
    %4247 = stablehlo.add %4242, %4246 : tensor<bf16>
    %4248 = stablehlo.convert %3163 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4249 = stablehlo.reduce(%4248 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4250 = "stablehlo.all_reduce"(%4249) <{channel_handle = #stablehlo.channel_handle<handle = 225, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4251 = stablehlo.convert %4250 : (tensor<f32>) -> tensor<bf16>
    %4252 = stablehlo.add %4247, %4251 : tensor<bf16>
    %4253 = stablehlo.convert %3181 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4254 = stablehlo.reduce(%4253 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4255 = "stablehlo.all_reduce"(%4254) <{channel_handle = #stablehlo.channel_handle<handle = 226, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4256 = stablehlo.convert %4255 : (tensor<f32>) -> tensor<bf16>
    %4257 = stablehlo.add %4252, %4256 : tensor<bf16>
    %4258 = stablehlo.convert %3199 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4259 = stablehlo.reduce(%4258 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4260 = "stablehlo.all_reduce"(%4259) <{channel_handle = #stablehlo.channel_handle<handle = 227, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4261 = stablehlo.convert %4260 : (tensor<f32>) -> tensor<bf16>
    %4262 = stablehlo.add %4257, %4261 : tensor<bf16>
    %4263 = stablehlo.convert %3217 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4264 = stablehlo.reduce(%4263 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4265 = "stablehlo.all_reduce"(%4264) <{channel_handle = #stablehlo.channel_handle<handle = 228, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4266 = stablehlo.convert %4265 : (tensor<f32>) -> tensor<bf16>
    %4267 = stablehlo.add %4262, %4266 : tensor<bf16>
    %4268 = stablehlo.convert %3235 : (tensor<576xbf16>) -> tensor<576xf32>
    %4269 = stablehlo.reduce(%4268 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4270 = stablehlo.convert %4269 : (tensor<f32>) -> tensor<bf16>
    %4271 = stablehlo.add %4267, %4270 : tensor<bf16>
    %4272 = stablehlo.convert %3253 : (tensor<576xbf16>) -> tensor<576xf32>
    %4273 = stablehlo.reduce(%4272 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4274 = stablehlo.convert %4273 : (tensor<f32>) -> tensor<bf16>
    %4275 = stablehlo.add %4271, %4274 : tensor<bf16>
    %4276 = stablehlo.convert %3271 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4277 = stablehlo.reduce(%4276 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4278 = "stablehlo.all_reduce"(%4277) <{channel_handle = #stablehlo.channel_handle<handle = 229, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4279 = stablehlo.convert %4278 : (tensor<f32>) -> tensor<bf16>
    %4280 = stablehlo.add %4275, %4279 : tensor<bf16>
    %4281 = stablehlo.convert %3289 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4282 = stablehlo.reduce(%4281 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4283 = "stablehlo.all_reduce"(%4282) <{channel_handle = #stablehlo.channel_handle<handle = 230, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4284 = stablehlo.convert %4283 : (tensor<f32>) -> tensor<bf16>
    %4285 = stablehlo.add %4280, %4284 : tensor<bf16>
    %4286 = stablehlo.convert %3307 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4287 = stablehlo.reduce(%4286 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4288 = "stablehlo.all_reduce"(%4287) <{channel_handle = #stablehlo.channel_handle<handle = 231, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4289 = stablehlo.convert %4288 : (tensor<f32>) -> tensor<bf16>
    %4290 = stablehlo.add %4285, %4289 : tensor<bf16>
    %4291 = stablehlo.convert %3325 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4292 = stablehlo.reduce(%4291 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4293 = "stablehlo.all_reduce"(%4292) <{channel_handle = #stablehlo.channel_handle<handle = 232, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4294 = stablehlo.convert %4293 : (tensor<f32>) -> tensor<bf16>
    %4295 = stablehlo.add %4290, %4294 : tensor<bf16>
    %4296 = stablehlo.convert %3343 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4297 = stablehlo.reduce(%4296 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4298 = "stablehlo.all_reduce"(%4297) <{channel_handle = #stablehlo.channel_handle<handle = 233, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4299 = stablehlo.convert %4298 : (tensor<f32>) -> tensor<bf16>
    %4300 = stablehlo.add %4295, %4299 : tensor<bf16>
    %4301 = stablehlo.convert %3361 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4302 = stablehlo.reduce(%4301 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4303 = "stablehlo.all_reduce"(%4302) <{channel_handle = #stablehlo.channel_handle<handle = 234, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4304 = stablehlo.convert %4303 : (tensor<f32>) -> tensor<bf16>
    %4305 = stablehlo.add %4300, %4304 : tensor<bf16>
    %4306 = stablehlo.convert %3379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4307 = stablehlo.reduce(%4306 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4308 = "stablehlo.all_reduce"(%4307) <{channel_handle = #stablehlo.channel_handle<handle = 235, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4309 = stablehlo.convert %4308 : (tensor<f32>) -> tensor<bf16>
    %4310 = stablehlo.add %4305, %4309 : tensor<bf16>
    %4311 = stablehlo.convert %3397 : (tensor<576xbf16>) -> tensor<576xf32>
    %4312 = stablehlo.reduce(%4311 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4313 = stablehlo.convert %4312 : (tensor<f32>) -> tensor<bf16>
    %4314 = stablehlo.add %4310, %4313 : tensor<bf16>
    %4315 = stablehlo.convert %3415 : (tensor<576xbf16>) -> tensor<576xf32>
    %4316 = stablehlo.reduce(%4315 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4317 = stablehlo.convert %4316 : (tensor<f32>) -> tensor<bf16>
    %4318 = stablehlo.add %4314, %4317 : tensor<bf16>
    %4319 = stablehlo.convert %3433 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4320 = stablehlo.reduce(%4319 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4321 = "stablehlo.all_reduce"(%4320) <{channel_handle = #stablehlo.channel_handle<handle = 236, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4322 = stablehlo.convert %4321 : (tensor<f32>) -> tensor<bf16>
    %4323 = stablehlo.add %4318, %4322 : tensor<bf16>
    %4324 = stablehlo.convert %3451 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4325 = stablehlo.reduce(%4324 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4326 = "stablehlo.all_reduce"(%4325) <{channel_handle = #stablehlo.channel_handle<handle = 237, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4327 = stablehlo.convert %4326 : (tensor<f32>) -> tensor<bf16>
    %4328 = stablehlo.add %4323, %4327 : tensor<bf16>
    %4329 = stablehlo.convert %3469 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4330 = stablehlo.reduce(%4329 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4331 = "stablehlo.all_reduce"(%4330) <{channel_handle = #stablehlo.channel_handle<handle = 238, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4332 = stablehlo.convert %4331 : (tensor<f32>) -> tensor<bf16>
    %4333 = stablehlo.add %4328, %4332 : tensor<bf16>
    %4334 = stablehlo.convert %3487 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4335 = stablehlo.reduce(%4334 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4336 = "stablehlo.all_reduce"(%4335) <{channel_handle = #stablehlo.channel_handle<handle = 239, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4337 = stablehlo.convert %4336 : (tensor<f32>) -> tensor<bf16>
    %4338 = stablehlo.add %4333, %4337 : tensor<bf16>
    %4339 = stablehlo.convert %3505 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4340 = stablehlo.reduce(%4339 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4341 = "stablehlo.all_reduce"(%4340) <{channel_handle = #stablehlo.channel_handle<handle = 240, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4342 = stablehlo.convert %4341 : (tensor<f32>) -> tensor<bf16>
    %4343 = stablehlo.add %4338, %4342 : tensor<bf16>
    %4344 = stablehlo.convert %3523 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4345 = stablehlo.reduce(%4344 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4346 = "stablehlo.all_reduce"(%4345) <{channel_handle = #stablehlo.channel_handle<handle = 241, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4347 = stablehlo.convert %4346 : (tensor<f32>) -> tensor<bf16>
    %4348 = stablehlo.add %4343, %4347 : tensor<bf16>
    %4349 = stablehlo.convert %3541 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4350 = stablehlo.reduce(%4349 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4351 = "stablehlo.all_reduce"(%4350) <{channel_handle = #stablehlo.channel_handle<handle = 242, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4352 = stablehlo.convert %4351 : (tensor<f32>) -> tensor<bf16>
    %4353 = stablehlo.add %4348, %4352 : tensor<bf16>
    %4354 = stablehlo.convert %3559 : (tensor<576xbf16>) -> tensor<576xf32>
    %4355 = stablehlo.reduce(%4354 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4356 = stablehlo.convert %4355 : (tensor<f32>) -> tensor<bf16>
    %4357 = stablehlo.add %4353, %4356 : tensor<bf16>
    %4358 = stablehlo.convert %3577 : (tensor<576xbf16>) -> tensor<576xf32>
    %4359 = stablehlo.reduce(%4358 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4360 = stablehlo.convert %4359 : (tensor<f32>) -> tensor<bf16>
    %4361 = stablehlo.add %4357, %4360 : tensor<bf16>
    %4362 = stablehlo.convert %3595 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4363 = stablehlo.reduce(%4362 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4364 = "stablehlo.all_reduce"(%4363) <{channel_handle = #stablehlo.channel_handle<handle = 243, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4365 = stablehlo.convert %4364 : (tensor<f32>) -> tensor<bf16>
    %4366 = stablehlo.add %4361, %4365 : tensor<bf16>
    %4367 = stablehlo.convert %3613 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4368 = stablehlo.reduce(%4367 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4369 = "stablehlo.all_reduce"(%4368) <{channel_handle = #stablehlo.channel_handle<handle = 244, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4370 = stablehlo.convert %4369 : (tensor<f32>) -> tensor<bf16>
    %4371 = stablehlo.add %4366, %4370 : tensor<bf16>
    %4372 = stablehlo.convert %3631 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4373 = stablehlo.reduce(%4372 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4374 = "stablehlo.all_reduce"(%4373) <{channel_handle = #stablehlo.channel_handle<handle = 245, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4375 = stablehlo.convert %4374 : (tensor<f32>) -> tensor<bf16>
    %4376 = stablehlo.add %4371, %4375 : tensor<bf16>
    %4377 = stablehlo.convert %3649 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4378 = stablehlo.reduce(%4377 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4379 = "stablehlo.all_reduce"(%4378) <{channel_handle = #stablehlo.channel_handle<handle = 246, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4380 = stablehlo.convert %4379 : (tensor<f32>) -> tensor<bf16>
    %4381 = stablehlo.add %4376, %4380 : tensor<bf16>
    %4382 = stablehlo.convert %3667 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4383 = stablehlo.reduce(%4382 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4384 = "stablehlo.all_reduce"(%4383) <{channel_handle = #stablehlo.channel_handle<handle = 247, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4385 = stablehlo.convert %4384 : (tensor<f32>) -> tensor<bf16>
    %4386 = stablehlo.add %4381, %4385 : tensor<bf16>
    %4387 = stablehlo.convert %3685 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4388 = stablehlo.reduce(%4387 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4389 = "stablehlo.all_reduce"(%4388) <{channel_handle = #stablehlo.channel_handle<handle = 248, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4390 = stablehlo.convert %4389 : (tensor<f32>) -> tensor<bf16>
    %4391 = stablehlo.add %4386, %4390 : tensor<bf16>
    %4392 = stablehlo.convert %3703 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4393 = stablehlo.reduce(%4392 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4394 = "stablehlo.all_reduce"(%4393) <{channel_handle = #stablehlo.channel_handle<handle = 249, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4395 = stablehlo.convert %4394 : (tensor<f32>) -> tensor<bf16>
    %4396 = stablehlo.add %4391, %4395 : tensor<bf16>
    %4397 = stablehlo.convert %3721 : (tensor<576xbf16>) -> tensor<576xf32>
    %4398 = stablehlo.reduce(%4397 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4399 = stablehlo.convert %4398 : (tensor<f32>) -> tensor<bf16>
    %4400 = stablehlo.add %4396, %4399 : tensor<bf16>
    %4401 = stablehlo.convert %3739 : (tensor<576xbf16>) -> tensor<576xf32>
    %4402 = stablehlo.reduce(%4401 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4403 = stablehlo.convert %4402 : (tensor<f32>) -> tensor<bf16>
    %4404 = stablehlo.add %4400, %4403 : tensor<bf16>
    %4405 = stablehlo.convert %3757 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4406 = stablehlo.reduce(%4405 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4407 = "stablehlo.all_reduce"(%4406) <{channel_handle = #stablehlo.channel_handle<handle = 250, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4408 = stablehlo.convert %4407 : (tensor<f32>) -> tensor<bf16>
    %4409 = stablehlo.add %4404, %4408 : tensor<bf16>
    %4410 = stablehlo.convert %3775 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4411 = stablehlo.reduce(%4410 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4412 = "stablehlo.all_reduce"(%4411) <{channel_handle = #stablehlo.channel_handle<handle = 251, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4413 = stablehlo.convert %4412 : (tensor<f32>) -> tensor<bf16>
    %4414 = stablehlo.add %4409, %4413 : tensor<bf16>
    %4415 = stablehlo.convert %3793 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4416 = stablehlo.reduce(%4415 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4417 = "stablehlo.all_reduce"(%4416) <{channel_handle = #stablehlo.channel_handle<handle = 252, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4418 = stablehlo.convert %4417 : (tensor<f32>) -> tensor<bf16>
    %4419 = stablehlo.add %4414, %4418 : tensor<bf16>
    %4420 = stablehlo.convert %3811 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4421 = stablehlo.reduce(%4420 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4422 = "stablehlo.all_reduce"(%4421) <{channel_handle = #stablehlo.channel_handle<handle = 253, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4423 = stablehlo.convert %4422 : (tensor<f32>) -> tensor<bf16>
    %4424 = stablehlo.add %4419, %4423 : tensor<bf16>
    %4425 = stablehlo.convert %3829 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4426 = stablehlo.reduce(%4425 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4427 = "stablehlo.all_reduce"(%4426) <{channel_handle = #stablehlo.channel_handle<handle = 254, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4428 = stablehlo.convert %4427 : (tensor<f32>) -> tensor<bf16>
    %4429 = stablehlo.add %4424, %4428 : tensor<bf16>
    %4430 = stablehlo.convert %3847 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4431 = stablehlo.reduce(%4430 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4432 = "stablehlo.all_reduce"(%4431) <{channel_handle = #stablehlo.channel_handle<handle = 255, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4433 = stablehlo.convert %4432 : (tensor<f32>) -> tensor<bf16>
    %4434 = stablehlo.add %4429, %4433 : tensor<bf16>
    %4435 = stablehlo.convert %3865 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4436 = stablehlo.reduce(%4435 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4437 = "stablehlo.all_reduce"(%4436) <{channel_handle = #stablehlo.channel_handle<handle = 256, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4438 = stablehlo.convert %4437 : (tensor<f32>) -> tensor<bf16>
    %4439 = stablehlo.add %4434, %4438 : tensor<bf16>
    %4440 = stablehlo.convert %3883 : (tensor<576xbf16>) -> tensor<576xf32>
    %4441 = stablehlo.reduce(%4440 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4442 = stablehlo.convert %4441 : (tensor<f32>) -> tensor<bf16>
    %4443 = stablehlo.add %4439, %4442 : tensor<bf16>
    %4444 = stablehlo.convert %3901 : (tensor<576xbf16>) -> tensor<576xf32>
    %4445 = stablehlo.reduce(%4444 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4446 = stablehlo.convert %4445 : (tensor<f32>) -> tensor<bf16>
    %4447 = stablehlo.add %4443, %4446 : tensor<bf16>
    %4448 = stablehlo.convert %3919 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4449 = stablehlo.reduce(%4448 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4450 = "stablehlo.all_reduce"(%4449) <{channel_handle = #stablehlo.channel_handle<handle = 257, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4451 = stablehlo.convert %4450 : (tensor<f32>) -> tensor<bf16>
    %4452 = stablehlo.add %4447, %4451 : tensor<bf16>
    %4453 = stablehlo.convert %3937 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4454 = stablehlo.reduce(%4453 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4455 = "stablehlo.all_reduce"(%4454) <{channel_handle = #stablehlo.channel_handle<handle = 258, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4456 = stablehlo.convert %4455 : (tensor<f32>) -> tensor<bf16>
    %4457 = stablehlo.add %4452, %4456 : tensor<bf16>
    %4458 = stablehlo.convert %3955 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4459 = stablehlo.reduce(%4458 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4460 = "stablehlo.all_reduce"(%4459) <{channel_handle = #stablehlo.channel_handle<handle = 259, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4461 = stablehlo.convert %4460 : (tensor<f32>) -> tensor<bf16>
    %4462 = stablehlo.add %4457, %4461 : tensor<bf16>
    %4463 = stablehlo.convert %3973 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4464 = stablehlo.reduce(%4463 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4465 = "stablehlo.all_reduce"(%4464) <{channel_handle = #stablehlo.channel_handle<handle = 260, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4466 = stablehlo.convert %4465 : (tensor<f32>) -> tensor<bf16>
    %4467 = stablehlo.add %4462, %4466 : tensor<bf16>
    %4468 = stablehlo.convert %3991 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4469 = stablehlo.reduce(%4468 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4470 = "stablehlo.all_reduce"(%4469) <{channel_handle = #stablehlo.channel_handle<handle = 261, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4471 = stablehlo.convert %4470 : (tensor<f32>) -> tensor<bf16>
    %4472 = stablehlo.add %4467, %4471 : tensor<bf16>
    %4473 = stablehlo.convert %4009 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,2016]{0,1}"} : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4474 = stablehlo.reduce(%4473 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4475 = "stablehlo.all_reduce"(%4474) <{channel_handle = #stablehlo.channel_handle<handle = 262, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4476 = stablehlo.convert %4475 : (tensor<f32>) -> tensor<bf16>
    %4477 = stablehlo.add %4472, %4476 : tensor<bf16>
    %4478 = stablehlo.convert %4027 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2016,144]{0,1}"} : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4479 = stablehlo.reduce(%4478 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4480 = "stablehlo.all_reduce"(%4479) <{channel_handle = #stablehlo.channel_handle<handle = 263, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4481 = stablehlo.convert %4480 : (tensor<f32>) -> tensor<bf16>
    %4482 = stablehlo.add %4477, %4481 : tensor<bf16>
    %4483 = stablehlo.convert %4045 : (tensor<576xbf16>) -> tensor<576xf32>
    %4484 = stablehlo.reduce(%4483 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4485 = stablehlo.convert %4484 : (tensor<f32>) -> tensor<bf16>
    %4486 = stablehlo.add %4482, %4485 : tensor<bf16>
    %4487 = stablehlo.convert %4063 : (tensor<576xbf16>) -> tensor<576xf32>
    %4488 = stablehlo.reduce(%4487 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4489 = stablehlo.convert %4488 : (tensor<f32>) -> tensor<bf16>
    %4490 = stablehlo.add %4486, %4489 : tensor<bf16>
    %4491 = stablehlo.convert %4081 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4492 = stablehlo.reduce(%4491 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4493 = "stablehlo.all_reduce"(%4492) <{channel_handle = #stablehlo.channel_handle<handle = 264, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4494 = stablehlo.convert %4493 : (tensor<f32>) -> tensor<bf16>
    %4495 = stablehlo.add %4490, %4494 : tensor<bf16>
    %4496 = stablehlo.convert %4099 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, xla_shape = "f32[9,128,144]{1,0,2}"} : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4497 = stablehlo.reduce(%4496 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4498 = "stablehlo.all_reduce"(%4497) <{channel_handle = #stablehlo.channel_handle<handle = 265, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4499 = stablehlo.convert %4498 : (tensor<f32>) -> tensor<bf16>
    %4500 = stablehlo.add %4495, %4499 : tensor<bf16>
    %4501 = stablehlo.convert %4117 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4502 = stablehlo.reduce(%4501 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4503 = "stablehlo.all_reduce"(%4502) <{channel_handle = #stablehlo.channel_handle<handle = 266, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4504 = stablehlo.convert %4503 : (tensor<f32>) -> tensor<bf16>
    %4505 = stablehlo.add %4500, %4504 : tensor<bf16>
    %4506 = stablehlo.convert %4135 {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[144,9,128]{0,2,1}"} : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4507 = stablehlo.reduce(%4506 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4508 = "stablehlo.all_reduce"(%4507) <{channel_handle = #stablehlo.channel_handle<handle = 267, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4509 = stablehlo.convert %4508 : (tensor<f32>) -> tensor<bf16>
    %4510 = stablehlo.add %4505, %4509 : tensor<bf16>
    %4511 = stablehlo.convert %4159 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[144,128256]{0,1}"} : (tensor<144x128256xbf16>) -> tensor<144x128256xf32>
    %4512 = stablehlo.reduce(%4511 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x128256xf32>, tensor<f32>) -> tensor<f32>
    %4513 = "stablehlo.all_reduce"(%4512) <{channel_handle = #stablehlo.channel_handle<handle = 268, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4514 = stablehlo.convert %4513 : (tensor<f32>) -> tensor<bf16>
    %4515 = stablehlo.add %4510, %4514 : tensor<bf16>
    %4516 = stablehlo.convert %4187 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128256,144]{0,1}"} : (tensor<128256x144xbf16>) -> tensor<128256x144xf32>
    %4517 = stablehlo.reduce(%4516 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<128256x144xf32>, tensor<f32>) -> tensor<f32>
    %4518 = "stablehlo.all_reduce"(%4517) <{channel_handle = #stablehlo.channel_handle<handle = 269, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4519 = stablehlo.convert %4518 : (tensor<f32>) -> tensor<bf16>
    %4520 = stablehlo.add %4515, %4519 : tensor<bf16>
    %4521 = stablehlo.sqrt %4520 : tensor<bf16>
    %4522 = stablehlo.broadcast_in_dim %803, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<1x4096x128256xf32>
    %4523 = stablehlo.subtract %800, %4522 : tensor<1x4096x128256xf32>
    %4524 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x4096x128256xf32>
    %4525 = stablehlo.select %814, %4523, %4524 : tensor<1x4096x128256xi1>, tensor<1x4096x128256xf32>
    %4526 = stablehlo.reduce(%4525 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x128256xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %4527 = stablehlo.negate %4526 : tensor<1x4096xf32>
    %4528 = stablehlo.multiply %804, %804 : tensor<1x4096xf32>
    %4529 = stablehlo.multiply %4528, %788 : tensor<1x4096xf32>
    %4530 = stablehlo.add %4527, %4529 : tensor<1x4096xf32>
    %4531 = mhlo.copy %4530 : tensor<1x4096xf32>
    %4532 = stablehlo.select %780, %4531, %788 : tensor<1x4096xi1>, tensor<1x4096xf32>
    %4533 = stablehlo.reduce(%4532 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<f32>
    %4534 = "stablehlo.all_reduce"(%4533) <{channel_handle = #stablehlo.channel_handle<handle = 270, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4535 = stablehlo.divide %4534, %785 : tensor<f32>
    %4536 = stablehlo.multiply %2973, %2973 : tensor<576xbf16>
    %4537 = stablehlo.convert %4536 : (tensor<576xbf16>) -> tensor<576xf32>
    %4538 = stablehlo.reduce(%4537 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4539 = stablehlo.convert %4538 : (tensor<f32>) -> tensor<bf16>
    %4540 = stablehlo.multiply %3001, %3001 : tensor<144x2016xbf16>
    %4541 = stablehlo.convert %4540 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4542 = stablehlo.reduce(%4541 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4543 = "stablehlo.all_reduce"(%4542) <{channel_handle = #stablehlo.channel_handle<handle = 271, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4544 = stablehlo.convert %4543 : (tensor<f32>) -> tensor<bf16>
    %4545 = stablehlo.add %4539, %4544 : tensor<bf16>
    %4546 = stablehlo.multiply %3019, %3019 : tensor<144x2016xbf16>
    %4547 = stablehlo.convert %4546 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4548 = stablehlo.reduce(%4547 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4549 = "stablehlo.all_reduce"(%4548) <{channel_handle = #stablehlo.channel_handle<handle = 272, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4550 = stablehlo.convert %4549 : (tensor<f32>) -> tensor<bf16>
    %4551 = stablehlo.add %4545, %4550 : tensor<bf16>
    %4552 = stablehlo.multiply %3047, %3047 : tensor<2016x144xbf16>
    %4553 = stablehlo.convert %4552 : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4554 = stablehlo.reduce(%4553 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4555 = "stablehlo.all_reduce"(%4554) <{channel_handle = #stablehlo.channel_handle<handle = 273, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4556 = stablehlo.convert %4555 : (tensor<f32>) -> tensor<bf16>
    %4557 = stablehlo.add %4551, %4556 : tensor<bf16>
    %4558 = stablehlo.multiply %3065, %3065 : tensor<576xbf16>
    %4559 = stablehlo.convert %4558 : (tensor<576xbf16>) -> tensor<576xf32>
    %4560 = stablehlo.reduce(%4559 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4561 = stablehlo.convert %4560 : (tensor<f32>) -> tensor<bf16>
    %4562 = stablehlo.add %4557, %4561 : tensor<bf16>
    %4563 = stablehlo.multiply %3083, %3083 : tensor<576xbf16>
    %4564 = stablehlo.convert %4563 : (tensor<576xbf16>) -> tensor<576xf32>
    %4565 = stablehlo.reduce(%4564 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4566 = stablehlo.convert %4565 : (tensor<f32>) -> tensor<bf16>
    %4567 = stablehlo.add %4562, %4566 : tensor<bf16>
    %4568 = stablehlo.multiply %3111, %3111 : tensor<144x9x128xbf16>
    %4569 = stablehlo.convert %4568 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4570 = stablehlo.reduce(%4569 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4571 = "stablehlo.all_reduce"(%4570) <{channel_handle = #stablehlo.channel_handle<handle = 274, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4572 = stablehlo.convert %4571 : (tensor<f32>) -> tensor<bf16>
    %4573 = stablehlo.add %4567, %4572 : tensor<bf16>
    %4574 = stablehlo.multiply %3139, %3139 : tensor<9x128x144xbf16>
    %4575 = stablehlo.convert %4574 : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4576 = stablehlo.reduce(%4575 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4577 = "stablehlo.all_reduce"(%4576) <{channel_handle = #stablehlo.channel_handle<handle = 275, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4578 = stablehlo.convert %4577 : (tensor<f32>) -> tensor<bf16>
    %4579 = stablehlo.add %4573, %4578 : tensor<bf16>
    %4580 = stablehlo.multiply %3157, %3157 : tensor<144x9x128xbf16>
    %4581 = stablehlo.convert %4580 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4582 = stablehlo.reduce(%4581 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4583 = "stablehlo.all_reduce"(%4582) <{channel_handle = #stablehlo.channel_handle<handle = 276, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4584 = stablehlo.convert %4583 : (tensor<f32>) -> tensor<bf16>
    %4585 = stablehlo.add %4579, %4584 : tensor<bf16>
    %4586 = stablehlo.multiply %3175, %3175 : tensor<144x9x128xbf16>
    %4587 = stablehlo.convert %4586 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4588 = stablehlo.reduce(%4587 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4589 = "stablehlo.all_reduce"(%4588) <{channel_handle = #stablehlo.channel_handle<handle = 277, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4590 = stablehlo.convert %4589 : (tensor<f32>) -> tensor<bf16>
    %4591 = stablehlo.add %4585, %4590 : tensor<bf16>
    %4592 = stablehlo.multiply %3193, %3193 : tensor<144x2016xbf16>
    %4593 = stablehlo.convert %4592 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4594 = stablehlo.reduce(%4593 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4595 = "stablehlo.all_reduce"(%4594) <{channel_handle = #stablehlo.channel_handle<handle = 278, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4596 = stablehlo.convert %4595 : (tensor<f32>) -> tensor<bf16>
    %4597 = stablehlo.add %4591, %4596 : tensor<bf16>
    %4598 = stablehlo.multiply %3211, %3211 : tensor<144x2016xbf16>
    %4599 = stablehlo.convert %4598 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4600 = stablehlo.reduce(%4599 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4601 = "stablehlo.all_reduce"(%4600) <{channel_handle = #stablehlo.channel_handle<handle = 279, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4602 = stablehlo.convert %4601 : (tensor<f32>) -> tensor<bf16>
    %4603 = stablehlo.add %4597, %4602 : tensor<bf16>
    %4604 = stablehlo.multiply %3229, %3229 : tensor<2016x144xbf16>
    %4605 = stablehlo.convert %4604 : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4606 = stablehlo.reduce(%4605 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4607 = "stablehlo.all_reduce"(%4606) <{channel_handle = #stablehlo.channel_handle<handle = 280, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4608 = stablehlo.convert %4607 : (tensor<f32>) -> tensor<bf16>
    %4609 = stablehlo.add %4603, %4608 : tensor<bf16>
    %4610 = stablehlo.multiply %3247, %3247 : tensor<576xbf16>
    %4611 = stablehlo.convert %4610 : (tensor<576xbf16>) -> tensor<576xf32>
    %4612 = stablehlo.reduce(%4611 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4613 = stablehlo.convert %4612 : (tensor<f32>) -> tensor<bf16>
    %4614 = stablehlo.add %4609, %4613 : tensor<bf16>
    %4615 = stablehlo.multiply %3265, %3265 : tensor<576xbf16>
    %4616 = stablehlo.convert %4615 : (tensor<576xbf16>) -> tensor<576xf32>
    %4617 = stablehlo.reduce(%4616 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4618 = stablehlo.convert %4617 : (tensor<f32>) -> tensor<bf16>
    %4619 = stablehlo.add %4614, %4618 : tensor<bf16>
    %4620 = stablehlo.multiply %3283, %3283 : tensor<144x9x128xbf16>
    %4621 = stablehlo.convert %4620 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4622 = stablehlo.reduce(%4621 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4623 = "stablehlo.all_reduce"(%4622) <{channel_handle = #stablehlo.channel_handle<handle = 281, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4624 = stablehlo.convert %4623 : (tensor<f32>) -> tensor<bf16>
    %4625 = stablehlo.add %4619, %4624 : tensor<bf16>
    %4626 = stablehlo.multiply %3301, %3301 : tensor<9x128x144xbf16>
    %4627 = stablehlo.convert %4626 : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4628 = stablehlo.reduce(%4627 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4629 = "stablehlo.all_reduce"(%4628) <{channel_handle = #stablehlo.channel_handle<handle = 282, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4630 = stablehlo.convert %4629 : (tensor<f32>) -> tensor<bf16>
    %4631 = stablehlo.add %4625, %4630 : tensor<bf16>
    %4632 = stablehlo.multiply %3319, %3319 : tensor<144x9x128xbf16>
    %4633 = stablehlo.convert %4632 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4634 = stablehlo.reduce(%4633 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4635 = "stablehlo.all_reduce"(%4634) <{channel_handle = #stablehlo.channel_handle<handle = 283, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4636 = stablehlo.convert %4635 : (tensor<f32>) -> tensor<bf16>
    %4637 = stablehlo.add %4631, %4636 : tensor<bf16>
    %4638 = stablehlo.multiply %3337, %3337 : tensor<144x9x128xbf16>
    %4639 = stablehlo.convert %4638 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4640 = stablehlo.reduce(%4639 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4641 = "stablehlo.all_reduce"(%4640) <{channel_handle = #stablehlo.channel_handle<handle = 284, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4642 = stablehlo.convert %4641 : (tensor<f32>) -> tensor<bf16>
    %4643 = stablehlo.add %4637, %4642 : tensor<bf16>
    %4644 = stablehlo.multiply %3355, %3355 : tensor<144x2016xbf16>
    %4645 = stablehlo.convert %4644 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4646 = stablehlo.reduce(%4645 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4647 = "stablehlo.all_reduce"(%4646) <{channel_handle = #stablehlo.channel_handle<handle = 285, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4648 = stablehlo.convert %4647 : (tensor<f32>) -> tensor<bf16>
    %4649 = stablehlo.add %4643, %4648 : tensor<bf16>
    %4650 = stablehlo.multiply %3373, %3373 : tensor<144x2016xbf16>
    %4651 = stablehlo.convert %4650 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4652 = stablehlo.reduce(%4651 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4653 = "stablehlo.all_reduce"(%4652) <{channel_handle = #stablehlo.channel_handle<handle = 286, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4654 = stablehlo.convert %4653 : (tensor<f32>) -> tensor<bf16>
    %4655 = stablehlo.add %4649, %4654 : tensor<bf16>
    %4656 = stablehlo.multiply %3391, %3391 : tensor<2016x144xbf16>
    %4657 = stablehlo.convert %4656 : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4658 = stablehlo.reduce(%4657 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4659 = "stablehlo.all_reduce"(%4658) <{channel_handle = #stablehlo.channel_handle<handle = 287, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4660 = stablehlo.convert %4659 : (tensor<f32>) -> tensor<bf16>
    %4661 = stablehlo.add %4655, %4660 : tensor<bf16>
    %4662 = stablehlo.multiply %3409, %3409 : tensor<576xbf16>
    %4663 = stablehlo.convert %4662 : (tensor<576xbf16>) -> tensor<576xf32>
    %4664 = stablehlo.reduce(%4663 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4665 = stablehlo.convert %4664 : (tensor<f32>) -> tensor<bf16>
    %4666 = stablehlo.add %4661, %4665 : tensor<bf16>
    %4667 = stablehlo.multiply %3427, %3427 : tensor<576xbf16>
    %4668 = stablehlo.convert %4667 : (tensor<576xbf16>) -> tensor<576xf32>
    %4669 = stablehlo.reduce(%4668 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4670 = stablehlo.convert %4669 : (tensor<f32>) -> tensor<bf16>
    %4671 = stablehlo.add %4666, %4670 : tensor<bf16>
    %4672 = stablehlo.multiply %3445, %3445 : tensor<144x9x128xbf16>
    %4673 = stablehlo.convert %4672 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4674 = stablehlo.reduce(%4673 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4675 = "stablehlo.all_reduce"(%4674) <{channel_handle = #stablehlo.channel_handle<handle = 288, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4676 = stablehlo.convert %4675 : (tensor<f32>) -> tensor<bf16>
    %4677 = stablehlo.add %4671, %4676 : tensor<bf16>
    %4678 = stablehlo.multiply %3463, %3463 : tensor<9x128x144xbf16>
    %4679 = stablehlo.convert %4678 : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4680 = stablehlo.reduce(%4679 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4681 = "stablehlo.all_reduce"(%4680) <{channel_handle = #stablehlo.channel_handle<handle = 289, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4682 = stablehlo.convert %4681 : (tensor<f32>) -> tensor<bf16>
    %4683 = stablehlo.add %4677, %4682 : tensor<bf16>
    %4684 = stablehlo.multiply %3481, %3481 : tensor<144x9x128xbf16>
    %4685 = stablehlo.convert %4684 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4686 = stablehlo.reduce(%4685 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4687 = "stablehlo.all_reduce"(%4686) <{channel_handle = #stablehlo.channel_handle<handle = 290, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4688 = stablehlo.convert %4687 : (tensor<f32>) -> tensor<bf16>
    %4689 = stablehlo.add %4683, %4688 : tensor<bf16>
    %4690 = stablehlo.multiply %3499, %3499 : tensor<144x9x128xbf16>
    %4691 = stablehlo.convert %4690 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4692 = stablehlo.reduce(%4691 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4693 = "stablehlo.all_reduce"(%4692) <{channel_handle = #stablehlo.channel_handle<handle = 291, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4694 = stablehlo.convert %4693 : (tensor<f32>) -> tensor<bf16>
    %4695 = stablehlo.add %4689, %4694 : tensor<bf16>
    %4696 = stablehlo.multiply %3517, %3517 : tensor<144x2016xbf16>
    %4697 = stablehlo.convert %4696 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4698 = stablehlo.reduce(%4697 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4699 = "stablehlo.all_reduce"(%4698) <{channel_handle = #stablehlo.channel_handle<handle = 292, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4700 = stablehlo.convert %4699 : (tensor<f32>) -> tensor<bf16>
    %4701 = stablehlo.add %4695, %4700 : tensor<bf16>
    %4702 = stablehlo.multiply %3535, %3535 : tensor<144x2016xbf16>
    %4703 = stablehlo.convert %4702 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4704 = stablehlo.reduce(%4703 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4705 = "stablehlo.all_reduce"(%4704) <{channel_handle = #stablehlo.channel_handle<handle = 293, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4706 = stablehlo.convert %4705 : (tensor<f32>) -> tensor<bf16>
    %4707 = stablehlo.add %4701, %4706 : tensor<bf16>
    %4708 = stablehlo.multiply %3553, %3553 : tensor<2016x144xbf16>
    %4709 = stablehlo.convert %4708 : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4710 = stablehlo.reduce(%4709 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4711 = "stablehlo.all_reduce"(%4710) <{channel_handle = #stablehlo.channel_handle<handle = 294, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4712 = stablehlo.convert %4711 : (tensor<f32>) -> tensor<bf16>
    %4713 = stablehlo.add %4707, %4712 : tensor<bf16>
    %4714 = stablehlo.multiply %3571, %3571 : tensor<576xbf16>
    %4715 = stablehlo.convert %4714 : (tensor<576xbf16>) -> tensor<576xf32>
    %4716 = stablehlo.reduce(%4715 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4717 = stablehlo.convert %4716 : (tensor<f32>) -> tensor<bf16>
    %4718 = stablehlo.add %4713, %4717 : tensor<bf16>
    %4719 = stablehlo.multiply %3589, %3589 : tensor<576xbf16>
    %4720 = stablehlo.convert %4719 : (tensor<576xbf16>) -> tensor<576xf32>
    %4721 = stablehlo.reduce(%4720 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4722 = stablehlo.convert %4721 : (tensor<f32>) -> tensor<bf16>
    %4723 = stablehlo.add %4718, %4722 : tensor<bf16>
    %4724 = stablehlo.multiply %3607, %3607 : tensor<144x9x128xbf16>
    %4725 = stablehlo.convert %4724 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4726 = stablehlo.reduce(%4725 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4727 = "stablehlo.all_reduce"(%4726) <{channel_handle = #stablehlo.channel_handle<handle = 295, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4728 = stablehlo.convert %4727 : (tensor<f32>) -> tensor<bf16>
    %4729 = stablehlo.add %4723, %4728 : tensor<bf16>
    %4730 = stablehlo.multiply %3625, %3625 : tensor<9x128x144xbf16>
    %4731 = stablehlo.convert %4730 : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4732 = stablehlo.reduce(%4731 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4733 = "stablehlo.all_reduce"(%4732) <{channel_handle = #stablehlo.channel_handle<handle = 296, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4734 = stablehlo.convert %4733 : (tensor<f32>) -> tensor<bf16>
    %4735 = stablehlo.add %4729, %4734 : tensor<bf16>
    %4736 = stablehlo.multiply %3643, %3643 : tensor<144x9x128xbf16>
    %4737 = stablehlo.convert %4736 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4738 = stablehlo.reduce(%4737 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4739 = "stablehlo.all_reduce"(%4738) <{channel_handle = #stablehlo.channel_handle<handle = 297, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4740 = stablehlo.convert %4739 : (tensor<f32>) -> tensor<bf16>
    %4741 = stablehlo.add %4735, %4740 : tensor<bf16>
    %4742 = stablehlo.multiply %3661, %3661 : tensor<144x9x128xbf16>
    %4743 = stablehlo.convert %4742 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4744 = stablehlo.reduce(%4743 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4745 = "stablehlo.all_reduce"(%4744) <{channel_handle = #stablehlo.channel_handle<handle = 298, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4746 = stablehlo.convert %4745 : (tensor<f32>) -> tensor<bf16>
    %4747 = stablehlo.add %4741, %4746 : tensor<bf16>
    %4748 = stablehlo.multiply %3679, %3679 : tensor<144x2016xbf16>
    %4749 = stablehlo.convert %4748 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4750 = stablehlo.reduce(%4749 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4751 = "stablehlo.all_reduce"(%4750) <{channel_handle = #stablehlo.channel_handle<handle = 299, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4752 = stablehlo.convert %4751 : (tensor<f32>) -> tensor<bf16>
    %4753 = stablehlo.add %4747, %4752 : tensor<bf16>
    %4754 = stablehlo.multiply %3697, %3697 : tensor<144x2016xbf16>
    %4755 = stablehlo.convert %4754 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4756 = stablehlo.reduce(%4755 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4757 = "stablehlo.all_reduce"(%4756) <{channel_handle = #stablehlo.channel_handle<handle = 300, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4758 = stablehlo.convert %4757 : (tensor<f32>) -> tensor<bf16>
    %4759 = stablehlo.add %4753, %4758 : tensor<bf16>
    %4760 = stablehlo.multiply %3715, %3715 : tensor<2016x144xbf16>
    %4761 = stablehlo.convert %4760 : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4762 = stablehlo.reduce(%4761 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4763 = "stablehlo.all_reduce"(%4762) <{channel_handle = #stablehlo.channel_handle<handle = 301, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4764 = stablehlo.convert %4763 : (tensor<f32>) -> tensor<bf16>
    %4765 = stablehlo.add %4759, %4764 : tensor<bf16>
    %4766 = stablehlo.multiply %3733, %3733 : tensor<576xbf16>
    %4767 = stablehlo.convert %4766 : (tensor<576xbf16>) -> tensor<576xf32>
    %4768 = stablehlo.reduce(%4767 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4769 = stablehlo.convert %4768 : (tensor<f32>) -> tensor<bf16>
    %4770 = stablehlo.add %4765, %4769 : tensor<bf16>
    %4771 = stablehlo.multiply %3751, %3751 : tensor<576xbf16>
    %4772 = stablehlo.convert %4771 : (tensor<576xbf16>) -> tensor<576xf32>
    %4773 = stablehlo.reduce(%4772 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4774 = stablehlo.convert %4773 : (tensor<f32>) -> tensor<bf16>
    %4775 = stablehlo.add %4770, %4774 : tensor<bf16>
    %4776 = stablehlo.multiply %3769, %3769 : tensor<144x9x128xbf16>
    %4777 = stablehlo.convert %4776 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4778 = stablehlo.reduce(%4777 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4779 = "stablehlo.all_reduce"(%4778) <{channel_handle = #stablehlo.channel_handle<handle = 302, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4780 = stablehlo.convert %4779 : (tensor<f32>) -> tensor<bf16>
    %4781 = stablehlo.add %4775, %4780 : tensor<bf16>
    %4782 = stablehlo.multiply %3787, %3787 : tensor<9x128x144xbf16>
    %4783 = stablehlo.convert %4782 : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4784 = stablehlo.reduce(%4783 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4785 = "stablehlo.all_reduce"(%4784) <{channel_handle = #stablehlo.channel_handle<handle = 303, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4786 = stablehlo.convert %4785 : (tensor<f32>) -> tensor<bf16>
    %4787 = stablehlo.add %4781, %4786 : tensor<bf16>
    %4788 = stablehlo.multiply %3805, %3805 : tensor<144x9x128xbf16>
    %4789 = stablehlo.convert %4788 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4790 = stablehlo.reduce(%4789 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4791 = "stablehlo.all_reduce"(%4790) <{channel_handle = #stablehlo.channel_handle<handle = 304, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4792 = stablehlo.convert %4791 : (tensor<f32>) -> tensor<bf16>
    %4793 = stablehlo.add %4787, %4792 : tensor<bf16>
    %4794 = stablehlo.multiply %3823, %3823 : tensor<144x9x128xbf16>
    %4795 = stablehlo.convert %4794 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4796 = stablehlo.reduce(%4795 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4797 = "stablehlo.all_reduce"(%4796) <{channel_handle = #stablehlo.channel_handle<handle = 305, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4798 = stablehlo.convert %4797 : (tensor<f32>) -> tensor<bf16>
    %4799 = stablehlo.add %4793, %4798 : tensor<bf16>
    %4800 = stablehlo.multiply %3841, %3841 : tensor<144x2016xbf16>
    %4801 = stablehlo.convert %4800 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4802 = stablehlo.reduce(%4801 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4803 = "stablehlo.all_reduce"(%4802) <{channel_handle = #stablehlo.channel_handle<handle = 306, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4804 = stablehlo.convert %4803 : (tensor<f32>) -> tensor<bf16>
    %4805 = stablehlo.add %4799, %4804 : tensor<bf16>
    %4806 = stablehlo.multiply %3859, %3859 : tensor<144x2016xbf16>
    %4807 = stablehlo.convert %4806 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4808 = stablehlo.reduce(%4807 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4809 = "stablehlo.all_reduce"(%4808) <{channel_handle = #stablehlo.channel_handle<handle = 307, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4810 = stablehlo.convert %4809 : (tensor<f32>) -> tensor<bf16>
    %4811 = stablehlo.add %4805, %4810 : tensor<bf16>
    %4812 = stablehlo.multiply %3877, %3877 : tensor<2016x144xbf16>
    %4813 = stablehlo.convert %4812 : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4814 = stablehlo.reduce(%4813 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4815 = "stablehlo.all_reduce"(%4814) <{channel_handle = #stablehlo.channel_handle<handle = 308, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4816 = stablehlo.convert %4815 : (tensor<f32>) -> tensor<bf16>
    %4817 = stablehlo.add %4811, %4816 : tensor<bf16>
    %4818 = stablehlo.multiply %3895, %3895 : tensor<576xbf16>
    %4819 = stablehlo.convert %4818 : (tensor<576xbf16>) -> tensor<576xf32>
    %4820 = stablehlo.reduce(%4819 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4821 = stablehlo.convert %4820 : (tensor<f32>) -> tensor<bf16>
    %4822 = stablehlo.add %4817, %4821 : tensor<bf16>
    %4823 = stablehlo.multiply %3913, %3913 : tensor<576xbf16>
    %4824 = stablehlo.convert %4823 : (tensor<576xbf16>) -> tensor<576xf32>
    %4825 = stablehlo.reduce(%4824 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4826 = stablehlo.convert %4825 : (tensor<f32>) -> tensor<bf16>
    %4827 = stablehlo.add %4822, %4826 : tensor<bf16>
    %4828 = stablehlo.multiply %3931, %3931 : tensor<144x9x128xbf16>
    %4829 = stablehlo.convert %4828 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4830 = stablehlo.reduce(%4829 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4831 = "stablehlo.all_reduce"(%4830) <{channel_handle = #stablehlo.channel_handle<handle = 309, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4832 = stablehlo.convert %4831 : (tensor<f32>) -> tensor<bf16>
    %4833 = stablehlo.add %4827, %4832 : tensor<bf16>
    %4834 = stablehlo.multiply %3949, %3949 : tensor<9x128x144xbf16>
    %4835 = stablehlo.convert %4834 : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4836 = stablehlo.reduce(%4835 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4837 = "stablehlo.all_reduce"(%4836) <{channel_handle = #stablehlo.channel_handle<handle = 310, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4838 = stablehlo.convert %4837 : (tensor<f32>) -> tensor<bf16>
    %4839 = stablehlo.add %4833, %4838 : tensor<bf16>
    %4840 = stablehlo.multiply %3967, %3967 : tensor<144x9x128xbf16>
    %4841 = stablehlo.convert %4840 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4842 = stablehlo.reduce(%4841 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4843 = "stablehlo.all_reduce"(%4842) <{channel_handle = #stablehlo.channel_handle<handle = 311, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4844 = stablehlo.convert %4843 : (tensor<f32>) -> tensor<bf16>
    %4845 = stablehlo.add %4839, %4844 : tensor<bf16>
    %4846 = stablehlo.multiply %3985, %3985 : tensor<144x9x128xbf16>
    %4847 = stablehlo.convert %4846 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4848 = stablehlo.reduce(%4847 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4849 = "stablehlo.all_reduce"(%4848) <{channel_handle = #stablehlo.channel_handle<handle = 312, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4850 = stablehlo.convert %4849 : (tensor<f32>) -> tensor<bf16>
    %4851 = stablehlo.add %4845, %4850 : tensor<bf16>
    %4852 = stablehlo.multiply %4003, %4003 : tensor<144x2016xbf16>
    %4853 = stablehlo.convert %4852 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4854 = stablehlo.reduce(%4853 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4855 = "stablehlo.all_reduce"(%4854) <{channel_handle = #stablehlo.channel_handle<handle = 313, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4856 = stablehlo.convert %4855 : (tensor<f32>) -> tensor<bf16>
    %4857 = stablehlo.add %4851, %4856 : tensor<bf16>
    %4858 = stablehlo.multiply %4021, %4021 : tensor<144x2016xbf16>
    %4859 = stablehlo.convert %4858 : (tensor<144x2016xbf16>) -> tensor<144x2016xf32>
    %4860 = stablehlo.reduce(%4859 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x2016xf32>, tensor<f32>) -> tensor<f32>
    %4861 = "stablehlo.all_reduce"(%4860) <{channel_handle = #stablehlo.channel_handle<handle = 314, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4862 = stablehlo.convert %4861 : (tensor<f32>) -> tensor<bf16>
    %4863 = stablehlo.add %4857, %4862 : tensor<bf16>
    %4864 = stablehlo.multiply %4039, %4039 : tensor<2016x144xbf16>
    %4865 = stablehlo.convert %4864 : (tensor<2016x144xbf16>) -> tensor<2016x144xf32>
    %4866 = stablehlo.reduce(%4865 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<2016x144xf32>, tensor<f32>) -> tensor<f32>
    %4867 = "stablehlo.all_reduce"(%4866) <{channel_handle = #stablehlo.channel_handle<handle = 315, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4868 = stablehlo.convert %4867 : (tensor<f32>) -> tensor<bf16>
    %4869 = stablehlo.add %4863, %4868 : tensor<bf16>
    %4870 = stablehlo.multiply %4057, %4057 : tensor<576xbf16>
    %4871 = stablehlo.convert %4870 : (tensor<576xbf16>) -> tensor<576xf32>
    %4872 = stablehlo.reduce(%4871 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4873 = stablehlo.convert %4872 : (tensor<f32>) -> tensor<bf16>
    %4874 = stablehlo.add %4869, %4873 : tensor<bf16>
    %4875 = stablehlo.multiply %4075, %4075 : tensor<576xbf16>
    %4876 = stablehlo.convert %4875 : (tensor<576xbf16>) -> tensor<576xf32>
    %4877 = stablehlo.reduce(%4876 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<576xf32>, tensor<f32>) -> tensor<f32>
    %4878 = stablehlo.convert %4877 : (tensor<f32>) -> tensor<bf16>
    %4879 = stablehlo.add %4874, %4878 : tensor<bf16>
    %4880 = stablehlo.multiply %4093, %4093 : tensor<144x9x128xbf16>
    %4881 = stablehlo.convert %4880 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4882 = stablehlo.reduce(%4881 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4883 = "stablehlo.all_reduce"(%4882) <{channel_handle = #stablehlo.channel_handle<handle = 316, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4884 = stablehlo.convert %4883 : (tensor<f32>) -> tensor<bf16>
    %4885 = stablehlo.add %4879, %4884 : tensor<bf16>
    %4886 = stablehlo.multiply %4111, %4111 : tensor<9x128x144xbf16>
    %4887 = stablehlo.convert %4886 : (tensor<9x128x144xbf16>) -> tensor<9x128x144xf32>
    %4888 = stablehlo.reduce(%4887 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<9x128x144xf32>, tensor<f32>) -> tensor<f32>
    %4889 = "stablehlo.all_reduce"(%4888) <{channel_handle = #stablehlo.channel_handle<handle = 317, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4890 = stablehlo.convert %4889 : (tensor<f32>) -> tensor<bf16>
    %4891 = stablehlo.add %4885, %4890 : tensor<bf16>
    %4892 = stablehlo.multiply %4129, %4129 : tensor<144x9x128xbf16>
    %4893 = stablehlo.convert %4892 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4894 = stablehlo.reduce(%4893 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4895 = "stablehlo.all_reduce"(%4894) <{channel_handle = #stablehlo.channel_handle<handle = 318, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4896 = stablehlo.convert %4895 : (tensor<f32>) -> tensor<bf16>
    %4897 = stablehlo.add %4891, %4896 : tensor<bf16>
    %4898 = stablehlo.multiply %4147, %4147 : tensor<144x9x128xbf16>
    %4899 = stablehlo.convert %4898 : (tensor<144x9x128xbf16>) -> tensor<144x9x128xf32>
    %4900 = stablehlo.reduce(%4899 init: %cst_11) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<144x9x128xf32>, tensor<f32>) -> tensor<f32>
    %4901 = "stablehlo.all_reduce"(%4900) <{channel_handle = #stablehlo.channel_handle<handle = 319, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4902 = stablehlo.convert %4901 : (tensor<f32>) -> tensor<bf16>
    %4903 = stablehlo.add %4897, %4902 : tensor<bf16>
    %4904 = stablehlo.multiply %4175, %4175 : tensor<144x128256xbf16>
    %4905 = stablehlo.convert %4904 : (tensor<144x128256xbf16>) -> tensor<144x128256xf32>
    %4906 = stablehlo.reduce(%4905 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<144x128256xf32>, tensor<f32>) -> tensor<f32>
    %4907 = "stablehlo.all_reduce"(%4906) <{channel_handle = #stablehlo.channel_handle<handle = 320, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4908 = stablehlo.convert %4907 : (tensor<f32>) -> tensor<bf16>
    %4909 = stablehlo.add %4903, %4908 : tensor<bf16>
    %4910 = stablehlo.multiply %4203, %4203 : tensor<128256x144xbf16>
    %4911 = stablehlo.convert %4910 : (tensor<128256x144xbf16>) -> tensor<128256x144xf32>
    %4912 = stablehlo.reduce(%4911 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<128256x144xf32>, tensor<f32>) -> tensor<f32>
    %4913 = "stablehlo.all_reduce"(%4912) <{channel_handle = #stablehlo.channel_handle<handle = 321, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4914 = stablehlo.convert %4913 : (tensor<f32>) -> tensor<bf16>
    %4915 = stablehlo.add %4909, %4914 : tensor<bf16>
    %4916 = stablehlo.sqrt %4915 : tensor<bf16>
    %4917 = "stablehlo.all_reduce"(%2084) <{channel_handle = #stablehlo.channel_handle<handle = 322, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4918 = stablehlo.convert %4917 : (tensor<f32>) -> tensor<bf16>
    %4919 = stablehlo.add %827, %4918 : tensor<bf16>
    %4920 = "stablehlo.all_reduce"(%2095) <{channel_handle = #stablehlo.channel_handle<handle = 323, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4921 = stablehlo.convert %4920 : (tensor<f32>) -> tensor<bf16>
    %4922 = stablehlo.add %4919, %4921 : tensor<bf16>
    %4923 = "stablehlo.all_reduce"(%2106) <{channel_handle = #stablehlo.channel_handle<handle = 324, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4924 = stablehlo.convert %4923 : (tensor<f32>) -> tensor<bf16>
    %4925 = stablehlo.add %4922, %4924 : tensor<bf16>
    %4926 = stablehlo.add %4925, %2122 : tensor<bf16>
    %4927 = stablehlo.add %4926, %2275 : tensor<bf16>
    %4928 = "stablehlo.all_reduce"(%2287) <{channel_handle = #stablehlo.channel_handle<handle = 325, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4929 = stablehlo.convert %4928 : (tensor<f32>) -> tensor<bf16>
    %4930 = stablehlo.add %4927, %4929 : tensor<bf16>
    %4931 = "stablehlo.all_reduce"(%2299) <{channel_handle = #stablehlo.channel_handle<handle = 326, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4932 = stablehlo.convert %4931 : (tensor<f32>) -> tensor<bf16>
    %4933 = stablehlo.add %4930, %4932 : tensor<bf16>
    %4934 = "stablehlo.all_reduce"(%2309) <{channel_handle = #stablehlo.channel_handle<handle = 327, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4935 = stablehlo.convert %4934 : (tensor<f32>) -> tensor<bf16>
    %4936 = stablehlo.add %4933, %4935 : tensor<bf16>
    %4937 = "stablehlo.all_reduce"(%2319) <{channel_handle = #stablehlo.channel_handle<handle = 328, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4938 = stablehlo.convert %4937 : (tensor<f32>) -> tensor<bf16>
    %4939 = stablehlo.add %4936, %4938 : tensor<bf16>
    %4940 = "stablehlo.all_reduce"(%2332) <{channel_handle = #stablehlo.channel_handle<handle = 329, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4941 = stablehlo.convert %4940 : (tensor<f32>) -> tensor<bf16>
    %4942 = stablehlo.add %4939, %4941 : tensor<bf16>
    %4943 = "stablehlo.all_reduce"(%2341) <{channel_handle = #stablehlo.channel_handle<handle = 330, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4944 = stablehlo.convert %4943 : (tensor<f32>) -> tensor<bf16>
    %4945 = stablehlo.add %4942, %4944 : tensor<bf16>
    %4946 = "stablehlo.all_reduce"(%2352) <{channel_handle = #stablehlo.channel_handle<handle = 331, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4947 = stablehlo.convert %4946 : (tensor<f32>) -> tensor<bf16>
    %4948 = stablehlo.add %4945, %4947 : tensor<bf16>
    %4949 = stablehlo.add %4948, %2362 : tensor<bf16>
    %4950 = stablehlo.add %4949, %2372 : tensor<bf16>
    %4951 = "stablehlo.all_reduce"(%2383) <{channel_handle = #stablehlo.channel_handle<handle = 332, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4952 = stablehlo.convert %4951 : (tensor<f32>) -> tensor<bf16>
    %4953 = stablehlo.add %4950, %4952 : tensor<bf16>
    %4954 = "stablehlo.all_reduce"(%2395) <{channel_handle = #stablehlo.channel_handle<handle = 333, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4955 = stablehlo.convert %4954 : (tensor<f32>) -> tensor<bf16>
    %4956 = stablehlo.add %4953, %4955 : tensor<bf16>
    %4957 = "stablehlo.all_reduce"(%2405) <{channel_handle = #stablehlo.channel_handle<handle = 334, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4958 = stablehlo.convert %4957 : (tensor<f32>) -> tensor<bf16>
    %4959 = stablehlo.add %4956, %4958 : tensor<bf16>
    %4960 = "stablehlo.all_reduce"(%2415) <{channel_handle = #stablehlo.channel_handle<handle = 335, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4961 = stablehlo.convert %4960 : (tensor<f32>) -> tensor<bf16>
    %4962 = stablehlo.add %4959, %4961 : tensor<bf16>
    %4963 = "stablehlo.all_reduce"(%2428) <{channel_handle = #stablehlo.channel_handle<handle = 336, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4964 = stablehlo.convert %4963 : (tensor<f32>) -> tensor<bf16>
    %4965 = stablehlo.add %4962, %4964 : tensor<bf16>
    %4966 = "stablehlo.all_reduce"(%2437) <{channel_handle = #stablehlo.channel_handle<handle = 337, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4967 = stablehlo.convert %4966 : (tensor<f32>) -> tensor<bf16>
    %4968 = stablehlo.add %4965, %4967 : tensor<bf16>
    %4969 = "stablehlo.all_reduce"(%2448) <{channel_handle = #stablehlo.channel_handle<handle = 338, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4970 = stablehlo.convert %4969 : (tensor<f32>) -> tensor<bf16>
    %4971 = stablehlo.add %4968, %4970 : tensor<bf16>
    %4972 = stablehlo.add %4971, %2458 : tensor<bf16>
    %4973 = stablehlo.add %4972, %2468 : tensor<bf16>
    %4974 = "stablehlo.all_reduce"(%2479) <{channel_handle = #stablehlo.channel_handle<handle = 339, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4975 = stablehlo.convert %4974 : (tensor<f32>) -> tensor<bf16>
    %4976 = stablehlo.add %4973, %4975 : tensor<bf16>
    %4977 = "stablehlo.all_reduce"(%2491) <{channel_handle = #stablehlo.channel_handle<handle = 340, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4978 = stablehlo.convert %4977 : (tensor<f32>) -> tensor<bf16>
    %4979 = stablehlo.add %4976, %4978 : tensor<bf16>
    %4980 = "stablehlo.all_reduce"(%2501) <{channel_handle = #stablehlo.channel_handle<handle = 341, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4981 = stablehlo.convert %4980 : (tensor<f32>) -> tensor<bf16>
    %4982 = stablehlo.add %4979, %4981 : tensor<bf16>
    %4983 = "stablehlo.all_reduce"(%2511) <{channel_handle = #stablehlo.channel_handle<handle = 342, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4984 = stablehlo.convert %4983 : (tensor<f32>) -> tensor<bf16>
    %4985 = stablehlo.add %4982, %4984 : tensor<bf16>
    %4986 = "stablehlo.all_reduce"(%2524) <{channel_handle = #stablehlo.channel_handle<handle = 343, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4987 = stablehlo.convert %4986 : (tensor<f32>) -> tensor<bf16>
    %4988 = stablehlo.add %4985, %4987 : tensor<bf16>
    %4989 = "stablehlo.all_reduce"(%2533) <{channel_handle = #stablehlo.channel_handle<handle = 344, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4990 = stablehlo.convert %4989 : (tensor<f32>) -> tensor<bf16>
    %4991 = stablehlo.add %4988, %4990 : tensor<bf16>
    %4992 = "stablehlo.all_reduce"(%2544) <{channel_handle = #stablehlo.channel_handle<handle = 345, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4993 = stablehlo.convert %4992 : (tensor<f32>) -> tensor<bf16>
    %4994 = stablehlo.add %4991, %4993 : tensor<bf16>
    %4995 = stablehlo.add %4994, %2554 : tensor<bf16>
    %4996 = stablehlo.add %4995, %2564 : tensor<bf16>
    %4997 = "stablehlo.all_reduce"(%2575) <{channel_handle = #stablehlo.channel_handle<handle = 346, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %4998 = stablehlo.convert %4997 : (tensor<f32>) -> tensor<bf16>
    %4999 = stablehlo.add %4996, %4998 : tensor<bf16>
    %5000 = "stablehlo.all_reduce"(%2587) <{channel_handle = #stablehlo.channel_handle<handle = 347, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5001 = stablehlo.convert %5000 : (tensor<f32>) -> tensor<bf16>
    %5002 = stablehlo.add %4999, %5001 : tensor<bf16>
    %5003 = "stablehlo.all_reduce"(%2597) <{channel_handle = #stablehlo.channel_handle<handle = 348, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5004 = stablehlo.convert %5003 : (tensor<f32>) -> tensor<bf16>
    %5005 = stablehlo.add %5002, %5004 : tensor<bf16>
    %5006 = "stablehlo.all_reduce"(%2607) <{channel_handle = #stablehlo.channel_handle<handle = 349, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5007 = stablehlo.convert %5006 : (tensor<f32>) -> tensor<bf16>
    %5008 = stablehlo.add %5005, %5007 : tensor<bf16>
    %5009 = "stablehlo.all_reduce"(%2620) <{channel_handle = #stablehlo.channel_handle<handle = 350, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5010 = stablehlo.convert %5009 : (tensor<f32>) -> tensor<bf16>
    %5011 = stablehlo.add %5008, %5010 : tensor<bf16>
    %5012 = "stablehlo.all_reduce"(%2629) <{channel_handle = #stablehlo.channel_handle<handle = 351, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5013 = stablehlo.convert %5012 : (tensor<f32>) -> tensor<bf16>
    %5014 = stablehlo.add %5011, %5013 : tensor<bf16>
    %5015 = "stablehlo.all_reduce"(%2640) <{channel_handle = #stablehlo.channel_handle<handle = 352, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5016 = stablehlo.convert %5015 : (tensor<f32>) -> tensor<bf16>
    %5017 = stablehlo.add %5014, %5016 : tensor<bf16>
    %5018 = stablehlo.add %5017, %2650 : tensor<bf16>
    %5019 = stablehlo.add %5018, %2660 : tensor<bf16>
    %5020 = "stablehlo.all_reduce"(%2671) <{channel_handle = #stablehlo.channel_handle<handle = 353, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5021 = stablehlo.convert %5020 : (tensor<f32>) -> tensor<bf16>
    %5022 = stablehlo.add %5019, %5021 : tensor<bf16>
    %5023 = "stablehlo.all_reduce"(%2683) <{channel_handle = #stablehlo.channel_handle<handle = 354, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5024 = stablehlo.convert %5023 : (tensor<f32>) -> tensor<bf16>
    %5025 = stablehlo.add %5022, %5024 : tensor<bf16>
    %5026 = "stablehlo.all_reduce"(%2693) <{channel_handle = #stablehlo.channel_handle<handle = 355, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5027 = stablehlo.convert %5026 : (tensor<f32>) -> tensor<bf16>
    %5028 = stablehlo.add %5025, %5027 : tensor<bf16>
    %5029 = "stablehlo.all_reduce"(%2703) <{channel_handle = #stablehlo.channel_handle<handle = 356, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5030 = stablehlo.convert %5029 : (tensor<f32>) -> tensor<bf16>
    %5031 = stablehlo.add %5028, %5030 : tensor<bf16>
    %5032 = "stablehlo.all_reduce"(%2716) <{channel_handle = #stablehlo.channel_handle<handle = 357, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5033 = stablehlo.convert %5032 : (tensor<f32>) -> tensor<bf16>
    %5034 = stablehlo.add %5031, %5033 : tensor<bf16>
    %5035 = "stablehlo.all_reduce"(%2725) <{channel_handle = #stablehlo.channel_handle<handle = 358, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5036 = stablehlo.convert %5035 : (tensor<f32>) -> tensor<bf16>
    %5037 = stablehlo.add %5034, %5036 : tensor<bf16>
    %5038 = "stablehlo.all_reduce"(%2736) <{channel_handle = #stablehlo.channel_handle<handle = 359, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5039 = stablehlo.convert %5038 : (tensor<f32>) -> tensor<bf16>
    %5040 = stablehlo.add %5037, %5039 : tensor<bf16>
    %5041 = stablehlo.add %5040, %2746 : tensor<bf16>
    %5042 = stablehlo.add %5041, %2756 : tensor<bf16>
    %5043 = "stablehlo.all_reduce"(%2767) <{channel_handle = #stablehlo.channel_handle<handle = 360, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5044 = stablehlo.convert %5043 : (tensor<f32>) -> tensor<bf16>
    %5045 = stablehlo.add %5042, %5044 : tensor<bf16>
    %5046 = "stablehlo.all_reduce"(%2779) <{channel_handle = #stablehlo.channel_handle<handle = 361, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5047 = stablehlo.convert %5046 : (tensor<f32>) -> tensor<bf16>
    %5048 = stablehlo.add %5045, %5047 : tensor<bf16>
    %5049 = "stablehlo.all_reduce"(%2789) <{channel_handle = #stablehlo.channel_handle<handle = 362, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5050 = stablehlo.convert %5049 : (tensor<f32>) -> tensor<bf16>
    %5051 = stablehlo.add %5048, %5050 : tensor<bf16>
    %5052 = "stablehlo.all_reduce"(%2799) <{channel_handle = #stablehlo.channel_handle<handle = 363, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5053 = stablehlo.convert %5052 : (tensor<f32>) -> tensor<bf16>
    %5054 = stablehlo.add %5051, %5053 : tensor<bf16>
    %5055 = "stablehlo.all_reduce"(%2812) <{channel_handle = #stablehlo.channel_handle<handle = 364, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5056 = stablehlo.convert %5055 : (tensor<f32>) -> tensor<bf16>
    %5057 = stablehlo.add %5054, %5056 : tensor<bf16>
    %5058 = "stablehlo.all_reduce"(%2821) <{channel_handle = #stablehlo.channel_handle<handle = 365, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5059 = stablehlo.convert %5058 : (tensor<f32>) -> tensor<bf16>
    %5060 = stablehlo.add %5057, %5059 : tensor<bf16>
    %5061 = "stablehlo.all_reduce"(%2832) <{channel_handle = #stablehlo.channel_handle<handle = 366, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5062 = stablehlo.convert %5061 : (tensor<f32>) -> tensor<bf16>
    %5063 = stablehlo.add %5060, %5062 : tensor<bf16>
    %5064 = stablehlo.add %5063, %2842 : tensor<bf16>
    %5065 = stablehlo.add %5064, %2852 : tensor<bf16>
    %5066 = "stablehlo.all_reduce"(%2863) <{channel_handle = #stablehlo.channel_handle<handle = 367, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5067 = stablehlo.convert %5066 : (tensor<f32>) -> tensor<bf16>
    %5068 = stablehlo.add %5065, %5067 : tensor<bf16>
    %5069 = "stablehlo.all_reduce"(%2875) <{channel_handle = #stablehlo.channel_handle<handle = 368, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5070 = stablehlo.convert %5069 : (tensor<f32>) -> tensor<bf16>
    %5071 = stablehlo.add %5068, %5070 : tensor<bf16>
    %5072 = "stablehlo.all_reduce"(%2885) <{channel_handle = #stablehlo.channel_handle<handle = 369, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5073 = stablehlo.convert %5072 : (tensor<f32>) -> tensor<bf16>
    %5074 = stablehlo.add %5071, %5073 : tensor<bf16>
    %5075 = "stablehlo.all_reduce"(%2895) <{channel_handle = #stablehlo.channel_handle<handle = 370, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5076 = stablehlo.convert %5075 : (tensor<f32>) -> tensor<bf16>
    %5077 = stablehlo.add %5074, %5076 : tensor<bf16>
    %5078 = "stablehlo.all_reduce"(%2904) <{channel_handle = #stablehlo.channel_handle<handle = 371, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5079 = stablehlo.convert %5078 : (tensor<f32>) -> tensor<bf16>
    %5080 = stablehlo.add %5077, %5079 : tensor<bf16>
    %5081 = "stablehlo.all_reduce"(%2931) <{channel_handle = #stablehlo.channel_handle<handle = 372, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg206: tensor<f32>, %arg207: tensor<f32>):
      %5085 = stablehlo.add %arg206, %arg207 : tensor<f32>
      stablehlo.return %5085 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5082 = stablehlo.convert %5081 : (tensor<f32>) -> tensor<bf16>
    %5083 = stablehlo.add %5080, %5082 : tensor<bf16>
    %5084 = stablehlo.sqrt %5083 : tensor<bf16>
    return %0, %2973, %3001, %3019, %3047, %3065, %3083, %3111, %3139, %3157, %3175, %3193, %3211, %3229, %3247, %3265, %3283, %3301, %3319, %3337, %3355, %3373, %3391, %3409, %3427, %3445, %3463, %3481, %3499, %3517, %3535, %3553, %3571, %3589, %3607, %3625, %3643, %3661, %3679, %3697, %3715, %3733, %3751, %3769, %3787, %3805, %3823, %3841, %3859, %3877, %3895, %3913, %3931, %3949, %3967, %3985, %4003, %4021, %4039, %4057, %4075, %4093, %4111, %4129, %4147, %4175, %4203, %2948, %2945, %2983, %3006, %3029, %3052, %3070, %3093, %3121, %3144, %3162, %3180, %3198, %3216, %3234, %3252, %3270, %3288, %3306, %3324, %3342, %3360, %3378, %3396, %3414, %3432, %3450, %3468, %3486, %3504, %3522, %3540, %3558, %3576, %3594, %3612, %3630, %3648, %3666, %3684, %3702, %3720, %3738, %3756, %3774, %3792, %3810, %3828, %3846, %3864, %3882, %3900, %3918, %3936, %3954, %3972, %3990, %4008, %4026, %4044, %4062, %4080, %4098, %4116, %4134, %4157, %4185, %2959, %2990, %3010, %3036, %3056, %3074, %3100, %3128, %3148, %3166, %3184, %3202, %3220, %3238, %3256, %3274, %3292, %3310, %3328, %3346, %3364, %3382, %3400, %3418, %3436, %3454, %3472, %3490, %3508, %3526, %3544, %3562, %3580, %3598, %3616, %3634, %3652, %3670, %3688, %3706, %3724, %3742, %3760, %3778, %3796, %3814, %3832, %3850, %3868, %3886, %3904, %3922, %3940, %3958, %3976, %3994, %4012, %4030, %4048, %4066, %4084, %4102, %4120, %4138, %4164, %4192, %4206, %4521, %4535, %cst_11, %cst_11, %4916, %5084, %783 : tensor<i32>, tensor<576xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x128256xbf16>, tensor<128256x144xbf16>, tensor<i32>, tensor<576xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x128256xbf16>, tensor<128256x144xbf16>, tensor<576xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x2016xbf16>, tensor<144x2016xbf16>, tensor<2016x144xbf16>, tensor<576xbf16>, tensor<576xbf16>, tensor<144x9x128xbf16>, tensor<9x128x144xbf16>, tensor<144x9x128xbf16>, tensor<144x9x128xbf16>, tensor<144x128256xbf16>, tensor<128256x144xbf16>, tensor<i32>, tensor<bf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<bf16>, tensor<bf16>, tensor<i32>
  }
}
