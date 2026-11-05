"""Precompute stain-normalised images for the Model 2 configs.

Usage: python scripts/precompute_stain.py

Reads every configs/*.yaml that sets data.stain_norm, and for each one caches the
deterministic part of its pipeline -- stain normalisation, pad-to-square, resize -- for
every split that config touches. Driven entirely by the configs, so the cache cannot
drift from the runs that use it: a config change with no cache rebuild fails loudly at
load time rather than training on stale images.

Idempotent: an image whose cached PNG already exists is skipped, so an interrupted run
resumes. Each file is written to a temporary name and renamed, so a half-written PNG is
never mistaken for a finished one.

LEAKAGE: this only touches the splits named in those configs, which are PBC. Target
domains are not cached here. Applying the same fixed, source-derived transform to target
images at final evaluation is required and legitimate, but it belongs to the eval
harness, where touching target data is a deliberate, once-only act.

Cost: roughly 7 minutes per method for Macenko and 15 for Reinhard over PBC, against
the same work repeated every epoch of every seed without it.
"""
import sys
import time
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def specs_from_config(config):
    """Every (dataset, split) a config reads, de-duplicated."""
    seen, out = set(), []
    candidates = [config.get("data", {})]
    candidates.append(config.get("train", {}).get("selection", {}))
    candidates += list(config.get("eval", {}).get("datasets", []))
    for spec in candidates:
        key = (spec.get("dataset"), spec.get("split"))
        if all(key) and key not in seen:
            seen.add(key)
            out.append({"dataset": key[0], "split": key[1]})
    return out


def main():
    sys.path.insert(0, str(ROOT))
    from src.data import build_dataset, load_attribute_space, load_label_space
    from src.data.cache import cache_dir, cached_path
    from src.data.transforms import build_deterministic_steps

    configs = []
    for path in sorted(CONFIGS.glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(config, dict) and config.get("data", {}).get("stain_norm"):
            configs.append((path, config))
    if not configs:
        sys.exit("no config sets data.stain_norm; nothing to precompute")

    labels, attributes = load_label_space(), load_attribute_space()
    started = time.time()
    total_written = 0

    for path, config in configs:
        method = config["data"]["stain_norm"]["method"]
        image_size = config["data"].get("image_size", 224)
        steps = build_deterministic_steps(image_size, {"method": method})
        print(f"\n{path.name}: {method} at {image_size}px -> "
              f"{cache_dir(method, image_size).relative_to(ROOT)}", flush=True)

        for spec in specs_from_config(config):
            # cache=False so build_dataset gives the raw dataset, not the cached wrapper
            # we are about to populate.
            source = build_dataset(
                {**spec, "image_size": image_size, "stain_norm": {"method": method, "cache": False}},
                labels=labels, attributes=attributes,
            )
            written = skipped = 0
            for i, record in enumerate(source.records, 1):
                destination = cached_path(method, image_size, record.path)
                if destination.exists():
                    skipped += 1
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(record.path) as image:
                    processed = steps(image.convert("RGB"))
                temporary = destination.with_suffix(".tmp.png")
                processed.save(temporary, format="PNG", optimize=False)
                temporary.replace(destination)
                written += 1
                if i % 500 == 0:
                    print(f"  {spec['dataset']}/{spec['split']}: {i}/{len(source.records)}",
                          flush=True)
            total_written += written
            print(f"  {spec['dataset']}/{spec['split']}: {written} written, {skipped} already cached",
                  flush=True)

    print(f"\n{total_written} images written in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
