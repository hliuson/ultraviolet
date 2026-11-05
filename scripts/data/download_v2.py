"""Download the extra raw sources for math-corpus-v2 into RAW2 (needs HF_TOKEN in env for the gated sets).

Usage: python -I download_v2.py RAW2_DIR [name=nfiles ...]
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download

SOURCES = {  # name: (repo, dir, prefix, default n files)
    "nemo4plus": ("nvidia/Nemotron-CC-Math-v1", "4plus", "", 2),
    "nemomind": ("nvidia/Nemotron-CC-Math-v1", "4plus_MIND", "", 1),
    "bigmath": ("SynthLabsAI/Big-Math-RL-Verified", "data", "", None),
}


def main():
    raw = sys.argv[1]
    want = {k: v[3] for k, v in SOURCES.items()}
    if len(sys.argv) > 2:  # explicit list: only these sources, n=-1 means all files
        want = {}
        for a in sys.argv[2:]:
            k, n = a.split("="); want[k] = None if int(n) < 0 else int(n)
    api = HfApi()
    mpath = os.path.join(raw, "download_manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    jobs = []
    for name, n in want.items():
        repo, d, prefix, _ = SOURCES[name]
        sha = api.dataset_info(repo).sha
        tree = [f for f in api.list_repo_tree(repo, repo_type="dataset", path_in_repo=d, recursive=True, revision=sha)
                if getattr(f, "size", None) is not None and f.path.endswith(".parquet")]
        tree.sort(key=lambda f: f.path)
        files = tree if n is None else tree[:n]
        manifest[name] = {"repo": repo, "sha": sha, "files": [f.path for f in files]}
        print(name, sha[:10], len(tree), "available;", [(f.path, round(f.size / 1e9, 2)) for f in files], flush=True)
        jobs += [(name, repo, sha, f.path) for f in files]
    json.dump(manifest, open(mpath, "w"), indent=1)

    def get(job):
        name, repo, sha, f = job
        hf_hub_download(repo, f, repo_type="dataset", revision=sha, local_dir=os.path.join(raw, name))
        return f

    with ThreadPoolExecutor(8) as ex:
        for i, f in enumerate(ex.map(get, jobs)):
            print(f"[{i+1}/{len(jobs)}] {f}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
