"""Record source bytes for a deployed release and its editable ByProt dependency."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform


def manifest(root, extensions):
    root = root.resolve()
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in extensions
            and not {"__pycache__", ".git", ".pytest_cache"}.intersection(p.relative_to(root).parts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {"python": platform.python_version(), "release": str(args.release.resolve()),
              "native_source": str(args.native_source.resolve()),
              "release_files": manifest(args.release, {".py", ".json", ".sh"}),
              "native_files": manifest(args.native_source, {".py"}),
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "peft", "byprot")}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
