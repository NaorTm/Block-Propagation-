# Real-World Adapter (Placeholder)

This folder contains scripts that ingest real node logs or RPC data and
convert them into a format comparable to the simulator outputs.

Supported input formats:

1) CSV
```
node_id,event,block_id,timestamp
```
- event: "header" or "block"
- timestamp: seconds (float)

2) Bitcoin Core debug.log
- Use `--format bitcoin-core`
- The parser extracts block hashes from lines containing `received block`, `new block`, or `UpdateTip: new best=...`

Outputs:
- Per-node arrival times CSV (node_id, block_id, first_block_time)
- Optional block-level summary CSV (t50/t90/t100 per block)
