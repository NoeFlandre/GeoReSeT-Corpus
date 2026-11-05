"""Thin Hugging Face Hub I/O layer."""

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def revision(repo: str) -> str:
    """Current commit of a dataset repo, pinned for the whole run."""
    return str(HfApi().dataset_info(repo).sha)


def list_files(repo: str, rev: str) -> frozenset[str]:
    return frozenset(HfApi().list_repo_files(repo, repo_type="dataset", revision=rev))


def download(repo: str, path: str, rev: str, dest: Path) -> Path:
    return Path(hf_hub_download(repo, path, repo_type="dataset", revision=rev, local_dir=dest))


def snapshot(repo: str, rev: str, dest: Path, patterns: list[str]) -> Path:
    """Download the files of ``repo`` at ``rev`` matching ``patterns`` into ``dest``."""
    return Path(
        snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=dest, allow_patterns=patterns)
    )


EXTRA_UNITS = ("unlabelled", "polygons", "scenes", "scene_photos", "assets")


def folder_commits(repo: str, rev: str, folders: tuple[str, ...]) -> dict[str, str]:
    """The last commit of each top-level folder of a dataset repo at ``rev``."""
    tree = HfApi().list_repo_tree(repo, repo_type="dataset", revision=rev, expand=True)
    return {e.path: e.last_commit.oid for e in tree if e.path in folders and e.last_commit is not None}


def publish_space(folder: Path, space: str) -> str:
    """Upload ``folder`` as the static Space ``space`` (created if missing), replacing its ``data/``."""
    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
    info = api.upload_folder(
        folder_path=folder,
        repo_id=space,
        repo_type="space",
        delete_patterns=["data/**"],
        commit_message="Explorer: data and app",
    )
    return str(info.commit_url)


def upload_units(folder: Path) -> list[str]:
    """Commit units: each data source, then unlabelled, polygons, scenes, assets; README last."""
    dirs = [*sorted(folder.glob("data/*")), *(folder / unit for unit in EXTRA_UNITS)]
    return [str(d.relative_to(folder)) for d in dirs if d.is_dir()]


def publish(folder: Path, repo_id: str, private: bool = False) -> str:
    """Create the dataset repo if needed and upload ``folder``, one commit per unit.

    Re-running is cheap: content already on the Hub is deduplicated and not re-sent.
    """
    api = HfApi()
    url = api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
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
