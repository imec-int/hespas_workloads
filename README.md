# Llama 3 size scaling 8 GPU 

Scaling the size of Llama3-like models on a single 8G GPU node.

# Operator Breakdown
```
10:39 $ hespas_mlir_analyzer ../workloads/llama3-100m.mlir -e
Parsing mlir_file: ../workloads/llama3-100m.mlir
MLIR Analysis Results:
-----------------------------------
Unique Operators   : 31

Operator Histogram:
  stablehlo.broadcast_in_dim....:  1744
  stablehlo.constant............:  1325
  stablehlo.multiply............:  1286
  stablehlo.convert.............:  1003
  stablehlo.add.................:   861
  stablehlo.divide..............:   433
  stablehlo.reduce..............:   421
  stablehlo.custom_call.........:   244
  stablehlo.dot_general.........:   208
  stablehlo.reshape.............:   136
  stablehlo.transpose...........:   135
  stablehlo.select..............:   104
  stablehlo.sqrt................:    70
  stablehlo.iota................:    58
  stablehlo.compare.............:    57
  stablehlo.slice...............:    56
  stablehlo.subtract............:    44
  stablehlo.negate..............:    43
  stablehlo.reduce_precision....:    35
  stablehlo.power...............:    30
  stablehlo.rsqrt...............:    29
  stablehlo.cosine..............:    29
  stablehlo.exponential.........:    29
  stablehlo.sine................:    28
  stablehlo.concatenate.........:    28
  stablehlo.and.................:    14
  stablehlo.optimization_barrier:     7
  stablehlo.log.................:     2
  stablehlo.maximum.............:     1
  stablehlo.minimum.............:     1
  func.return...................:     1
-----------------------------------
  Total Operators...............:  8462
```
