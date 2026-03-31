# Ingot

MoE inference engine for Apple Silicon. Runs massive Mixture-of-Experts language models on a MacBook by streaming expert weights from SSD.

**Status:** Phase 1 — Foundation (config, tokenizer, chat template)

## What It Does

Ingot exploits the sparsity of MoE models: a 397B parameter model only activates ~17B parameters per token. Expert weights are `mmap()`'d from NVMe SSD and managed by the macOS page cache — no custom caching layer, no fighting the OS.

## Supported Models

| Model | Total | Active | Experts |
|-------|-------|--------|---------|
| Qwen3.5-35B-A3B | 35B | 3B | 256 |
| Qwen3.5-122B-A10B | 122B | 10B | 256 |
| Qwen3.5-397B-A17B | 397B | 17B | 512 |

## Building

Requires macOS with Apple Silicon (Metal GPU).

```bash
make          # release build
make debug    # debug build with ASan/UBSan
make test     # run tests
make clean    # clean build artifacts
```

## Usage

```bash
# Tokenize text (available now)
./ingot tokenize --model /path/to/model --text "Hello world"

# Serve model (coming soon)
./ingot serve --model /path/to/model --port 8090
```

## API

OpenAI-compatible `/v1/chat/completions` with streaming, tool calling, and think tag filtering. See [SPEC.md](SPEC.md) for details.

## License

MIT
