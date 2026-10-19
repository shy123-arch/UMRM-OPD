"""Upload the release packages to ModelScope as PRIVATE repos (dry-run unless --yes).
  python tools/modelscope_upload.py --namespace <user_or_org> [--data DIR] [--ckpt DIR] [--code] [--yes]
Repos: <ns>/umrm-vho-data-v1197aug (dataset), <ns>/umrm-vho-ckpt (model; also carries a code snapshot under code/ when --code).
Auth: an existing `modelscope login` session or env MODELSCOPE_API_TOKEN (never printed / written by this script).
Licence: the clips are third-party web videos used for non-commercial research under application-based access;
the repos are created with visibility='private' and must stay private."""
import argparse, os, subprocess, sys, tempfile
from pathlib import Path

REL = Path("/mnt/workspace/Wilson/ck/umrm/release")


def size(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--namespace", required=True)
    ap.add_argument("--data", default=str(REL / "umrm_vho_data_v1197aug")); ap.add_argument("--ckpt", default=str(REL / "umrm_vho_ckpt"))
    ap.add_argument("--code", action="store_true"); ap.add_argument("--only", choices=["data", "ckpt"], default=None); ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    jobs = []
    if a.only in (None, "data"):
        jobs.append((f"{a.namespace}/umrm-vho-data-v1197aug", "dataset", a.data, ""))
    if a.only in (None, "ckpt"):
        jobs.append((f"{a.namespace}/umrm-vho-ckpt", "model", a.ckpt, ""))
        if a.code:
            repo = Path(__file__).resolve().parents[1]; tmp = Path(tempfile.mkdtemp()) / "code"
            subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "-o", str(tmp) + ".tar", "HEAD"], check=True)
            tmp.mkdir(); subprocess.run(["tar", "-xf", str(tmp) + ".tar", "-C", str(tmp)], check=True)
            jobs.append((f"{a.namespace}/umrm-vho-ckpt", "model", str(tmp), "code"))
    for rid, typ, d, sub in jobs:
        print(f"[{'UPLOAD' if a.yes else 'DRY-RUN'}] {typ:7s} {rid}  <- {d} ({size(d) / 2**30:.2f} GiB, {sum(1 for _ in Path(d).rglob('*') if _.is_file())} files)"
              + (f" into {sub}/" if sub else ""))
    if not a.yes:
        print("dry-run only; add --yes to create the PRIVATE repos and upload."); return
    from modelscope.hub.api import HubApi
    api = HubApi(); tok = os.environ.get("MODELSCOPE_API_TOKEN")
    if tok:
        api.login(tok)
    for rid, typ, d, sub in jobs:
        if not api.repo_exists(rid, repo_type=typ):
            api.create_repo(rid, repo_type=typ, visibility="private", license="Other", exist_ok=True)
        api.upload_folder(repo_id=rid, folder_path=d, path_in_repo=sub, repo_type=typ, commit_message=f"umrm-vho release: {Path(d).name}",
                          ignore_patterns=["*.tmp", "__pycache__/*"])
        print("uploaded", rid, sub or "/")


if __name__ == "__main__":
    main()
