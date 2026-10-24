"""Thin Hugging Face Hub I/O layer."""

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def revision(repo: str) -> str:
    """Current commit of a dataset repo, pinned for the whole run."""
    return str(HfApi().dataset_info(repo).sha)


def list_files(repo: str, rev: str) -> frozenset[str]:
    return frozenset(HfApi().list_repo_files(repo, repo_type="dataset", revision=rev))


def download(repo: str, path: str, rev: str, dest: Path) -> Path:
    return Path(hf_hub_download(repo, path, repo_type="dataset", revision=rev, local_dir=dest))


def upload_units(folder: Path) -> list[str]:
    """Commit units: each data source, then unlabelled, polygons, assets; README last."""
    dirs = [*sorted(folder.glob("data/*")), folder / "unlabelled", folder / "polygons", folder / "assets"]
    return [str(d.relative_to(folder)) for d in dirs if d.is_dir()]


def publish(folder: Path, repo_id: str) -> str:
    """Create the public dataset repo if needed and upload ``folder``, one commit per unit.

    Re-running is cheap: content already on the Hub is deduplicated and not re-sent.
    """
    api = HfApi()
    url = api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    for unit in upload_units(folder):
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=folder / unit,
            path_in_repo=unit,
            delete_patterns="*",
            commit_message=f"Upload {unit}",
        )
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=folder / "README.md",
        path_in_repo="README.md",
        commit_message="Update dataset card",
    )
    return str(url)
