# Reusable benchmark results

These files are copied from `ImageSearchatEdge` commit
`049f6384d7e80c11666701bb320a09727a7d8133` and are intentionally stored with
the NDP notebook so benchmark comparison does not depend on another GitHub
repository at runtime.

Included for all five benchmarks:

- `baseline`
- `v10`
- `v11`
- `v12`
- `edge_v1`
- `edge_v2`

The checked Edge result CSVs come from `ImageSearchatEdge` commit `6c79f16`
and make the completed runs directly inspectable on GitHub. The notebook does
not load those two saved versions into its comparison: it still generates
fresh Edge v1 and Edge v2 results from their portable vector exports.
