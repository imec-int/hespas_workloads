# Hespas Workloads

StableHLO workload traces for use with
[hespas](https://github.com/imec-int/hespas).

Each workload is stored on its own branch. The `main` branch is kept empty so
you can clone only the workloads you need without downloading the rest.

## Cloning a single workload

Use `--single-branch` and `--depth 1` to fetch only the branch you need:

```bash
git clone --branch llama-3-4 --single-branch --depth 1 \
  https://github.com/imec-int/hespas_workloads.git
```

This downloads only the `llama-3-4` workload with minimal history, keeping the
clone small and fast.

To fetch a different workload, replace `llama-3-4` with a different branch name.

## License

[MIT](LICENSE) — © imec
