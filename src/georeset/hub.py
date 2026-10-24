"""Thin Hugging Face Hub I/O layer."""

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def revision(repo: str) -> str:
    """Current commit of a dataset repo, pinned for the whole run."""
    return str(HfApi().dataset_info(repo).sha)


def list_files(repo: str, rev: str) -> frozenset[str]:
    return frozenset(HfApi().list_repo_files(repo, repo_type="dataset", revision=rev))


def download(repo: str, path: str, rev: str, dest: Path) -> Path:
    return Path(
        hf_hub_download(repo, path, repo_type="dataset", revision=rev, local_dir=dest)
    )
