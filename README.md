# Workloads repo

## Llama 3 4GPU

## Operator Breakdown
```
10:40 $ hespas_mlir_analyzer ../workloads/llama3-100m.mlir -e
Parsing mlir_file: ../workloads/llama3-100m.mlir
MLIR Analysis Results:
-----------------------------------
Unique Operators   : 35

Operator Histogram:
  stablehlo.multiply............:  1120
  stablehlo.add.................:   778
  stablehlo.convert.............:   607
  stablehlo.reduce..............:   311
  stablehlo.broadcast_in_dim....:   276
  stablehlo.divide..............:   273
  stablehlo.all_reduce..........:   272
  stablehlo.reshape.............:   231
  mhlo.copy.....................:   230
  stablehlo.dot_general.........:   208
  stablehlo.transpose...........:   112
  stablehlo.select..............:   108
  stablehlo.all_gather..........:   100
  stablehlo.sqrt................:    70
  stablehlo.slice...............:    56
  stablehlo.dynamic_slice.......:    52
  stablehlo.negate..............:    43
  stablehlo.subtract............:    42
  stablehlo.constant............:    38
  stablehlo.reduce_precision....:    35
  stablehlo.compare.............:    31
  stablehlo.rsqrt...............:    29
  stablehlo.exponential.........:    29
  stablehlo.concatenate.........:    28
  stablehlo.cosine..............:     9
  stablehlo.and.................:     8
  stablehlo.sine................:     8
  stablehlo.optimization_barrier:     7
  stablehlo.iota................:     5
  stablehlo.power...............:     3
  stablehlo.maximum.............:     1
  stablehlo.minimum.............:     1
  stablehlo.log.................:     1
  stablehlo.partition_id........:     1
  func.return...................:     1
-----------------------------------
  Total Operators...............:  5124
```
