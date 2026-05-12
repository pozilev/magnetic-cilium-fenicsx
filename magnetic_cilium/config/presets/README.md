Preset configs for common diploma/demo runs.

These files are intentionally lightweight YAML examples. They use the same
schema as `magnetic_cilium.config.schema` and can be copied or passed directly
to the CLI.

Examples:

```bash
python -m magnetic_cilium.cli.main validate magnetic_cilium/config/presets/final_mechanics.yaml
python -m magnetic_cilium.cli.main full magnetic_cilium/config/presets/debug_small.yaml --engine new --dry-run
```
