# ipasc_iqa

Benchmarking framework for Image Quality Assessment (IQA) of photoacoustic image reconstructions.

## Quick start

```bash
pip install ipasc_iqa
ipasc_iqa list measures
ipasc_iqa run --dataset path/to/dataset --measure psnr --measure ssim --output results.json
```

See [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) for the full design.
