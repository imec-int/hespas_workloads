# Llama2-7B scaling exports

Scaling of llama2-7B training from 4 to 128 GPUs on 4GPU nodes.
Most use FSDP within node except from firles labeled full-dp that only use data parallelism. Note that this export requires at least 107GB of HBM per GPU. 