"""Download the math-corpus-v1 raw parquet files (first N files per source) into RAW.

Usage: python -I download.py RAW_DIR
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download

# (name, repo, dir, filename prefix, n files)  n=None -> all
SOURCES = [
    ("finemath4", "HuggingFaceTB/finemath", "finemath-4plus", "train", 15),
    ("cosmo_automath", "HuggingFaceTB/cosmopedia", "data/auto_math_text", "train", 11),
    ("cosmo_khan", "HuggingFaceTB/cosmopedia", "data/khanacademy", "train", None),
    ("cosmo_openstax", "HuggingFaceTB/cosmopedia", "data/openstax", "train", None),
    ("numina_cot", "AI-MO/NuminaMath-CoT", "data", "train", None),
    ("numina15", "AI-MO/NuminaMath-1.5", "data", "train", None),
    ("omi2", "nvidia/OpenMathInstruct-2", "data", "train_1M", None),
    ("metamath", "meta-math/MetaMathQA", "", "MetaMathQA-395K", None),
    ("openr1", "open-r1/OpenR1-Math-220k", "data", "train", None),
    ("omr_cot", "nvidia/OpenMathReasoning", "data", "cot", 9),
    # eval sets (decontamination only)
    ("ev_math500", "HuggingFaceH4/MATH-500", "", "", None),
    ("ev_gsm8k", "openai/gsm8k", "main", "test", None),
    ("ev_aime24", "HuggingFaceH4/aime_2024", "", "", None),
    ("ev_aime25", "MathArena/aime_2025", "", "", None),
    ("ev_aime26", "MathArena/aime_2026", "", "", None),
    ("ev_amc23", "math-ai/amc23", "", "", None),
    ("ev_hmmt25", "MathArena/hmmt_feb_2025", "", "", None),
    ("ev_hmmt26", "MathArena/hmmt_feb_2026", "", "", None),
]
EXT = (".parquet", ".jsonl", ".json")


def main():
    raw = sys.argv[1]
    api = HfApi()
    jobs, manifest = [], {}
    for name, repo, d, prefix, n in SOURCES:
        sha = api.dataset_info(repo).sha
        tree = api.list_repo_tree(repo, repo_type="dataset", path_in_repo=d or None, recursive=True, revision=sha)
        files = sorted(f.path for f in tree if getattr(f, "size", None) is not None and f.path.endswith(EXT)
                       and os.path.basename(f.path).startswith(prefix) and not os.path.basename(f.path).startswith("."))
        if name.startswith("ev_"):
            files = [f for f in files if "train" not in os.path.basename(f) or name == "ev_aime24"]
        if prefix == "train" and name in ("numina15", "numina_cot"):
            files = [f for f in files if "test" not in f]
        files = files[:n] if n else files
        manifest[name] = {"repo": repo, "sha": sha, "files": files}
        jobs += [(name, repo, sha, f) for f in files]
        print(name, repo, sha[:10], len(files), files[:2], flush=True)
    mpath = os.path.join(raw, "download_manifest.json")
    old = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for k, v in old.items():
        if k not in manifest or not manifest[k]["files"]:
            manifest[k] = v
    json.dump(manifest, open(mpath, "w"), indent=1)

    def get(job):
        name, repo, sha, f = job
        out = os.path.join(raw, name)
        os.makedirs(out, exist_ok=True)
        hf_hub_download(repo, f, repo_type="dataset", revision=sha, local_dir=out)
        return f

    with ThreadPoolExecutor(16) as ex:
        for i, f in enumerate(ex.map(get, jobs)):
            print(f"[{i+1}/{len(jobs)}] {f}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
