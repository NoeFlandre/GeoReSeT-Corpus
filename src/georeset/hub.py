"""Thin Hugging Face Hub I/O layer."""

import shutil
from collections.abc import Iterable
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def revision(repo: str) -> str:
    """Current commit of a dataset repo, pinned for the whole run."""
    return str(HfApi().dataset_info(repo).sha)


def revision_of(repo: str, ref: str) -> str:
    """The commit a tag or branch of a dataset repo points to."""
    return str(HfApi().dataset_info(repo, revision=ref).sha)


def list_files(repo: str, rev: str) -> frozenset[str]:
    return frozenset(HfApi().list_repo_files(repo, repo_type="dataset", revision=rev))


def download(repo: str, path: str, rev: str, dest: Path) -> Path:
    return Path(hf_hub_download(repo, path, repo_type="dataset", revision=rev, local_dir=dest))


def snapshot(repo: str, rev: str, dest: Path, patterns: list[str]) -> Path:
    """Download the files of ``repo`` at ``rev`` matching ``patterns`` into ``dest``."""
    return Path(
        snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=dest, allow_patterns=patterns)
    )


EXTRA_UNITS = (
    "unlabelled", "polygons", "scenes", "scene_photos", "embeddings", "polygons_labels_shares",
    "embeddings_en", "assets",
)  # fmt: skip


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
    _upload(api, folder, repo_id, None)
    return str(url)


def _upload(api: HfApi, folder: Path, repo_id: str, revision: str | None) -> None:
    """One commit per unit of ``folder``, then the card, on ``revision`` (default: main)."""
    for unit in upload_units(folder):
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=folder / unit,
            path_in_repo=unit,
            delete_patterns="*",
            commit_message=f"Upload {unit}",
            revision=revision,
        )
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=folder / "README.md",
        path_in_repo="README.md",
        commit_message="Update dataset card",
        revision=revision,
    )


def propose(folder: Path, repo_id: str, title: str, drop: Iterable[str] = ()) -> str:
    """Upload ``folder`` to a new pull request of the dataset repo, for review; returns its URL.

    ``drop``: names in ``folder`` deleted first, so the pull request leaves them out (the release's own).
    """
    for name in drop:
        _delete(folder / name)
    api = HfApi()
    pr = api.create_pull_request(repo_id, title=title, repo_type="dataset")
    _upload(api, folder, repo_id, pr.git_reference)
    return str(pr.url)


def _delete(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def release(repo_id: str, number: int, tag_name: str) -> None:
    """Merge pull request ``number`` and tag the merged commit ``tag_name``, a release.

    Tags are never moved, so a name already taken stops the release before anything is merged. The Hub opens
    pull requests as drafts, which cannot be merged, so the draft is opened first.
    """
    api = HfApi()
    if tag_name in {tag.name for tag in api.list_repo_refs(repo_id, repo_type="dataset").tags}:
        raise ValueError(f"{repo_id} already has the tag {tag_name}; nothing was merged")
    api.change_discussion_status(repo_id, number, "open", repo_type="dataset")
    api.merge_pull_request(repo_id, number, repo_type="dataset")
    api.create_tag(repo_id, tag=tag_name, repo_type="dataset")
