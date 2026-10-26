from pathlib import Path

from georeset import hub


class _FakeApi:
    """Records every call with all its arguments."""

    def __init__(self):
        self.calls = []

    def create_repo(self, *args, **kwargs):
        self.calls.append(("create_repo", args, kwargs))
        return f"https://hf.co/datasets/{args[0]}"

    def upload_folder(self, *args, **kwargs):
        self.calls.append(("upload_folder", args, kwargs))

    def upload_file(self, *args, **kwargs):
        self.calls.append(("upload_file", args, kwargs))


def _out(tmp_path):
    for d in ["data/wikipedia/fra_Latn", "data/website/eng_Latn", "polygons", "assets", "unlabelled"]:
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "README.md").write_text("card")
    return tmp_path


def test_upload_units_order(tmp_path):
    assert hub.upload_units(_out(tmp_path)) == [
        "data/website",
        "data/wikipedia",
        "unlabelled",
        "polygons",
        "assets",
    ]


def test_upload_units_add_scenes_before_assets(tmp_path):
    out = _out(tmp_path)
    for d in ["scene_photos", "scenes"]:
        (out / d).mkdir()
    assert hub.upload_units(out)[-4:] == ["polygons", "scenes", "scene_photos", "assets"]


def test_upload_units_skip_missing_folders(tmp_path):
    (tmp_path / "data" / "website").mkdir(parents=True)
    assert hub.upload_units(tmp_path) == ["data/website"]


def test_publish_creates_public_repo_uploads_units_then_card(tmp_path, monkeypatch):
    api = _FakeApi()
    monkeypatch.setattr(hub, "HfApi", lambda: api)
    out = _out(tmp_path)
    assert hub.publish(out, "me/d") == "https://hf.co/datasets/me/d"
    assert api.calls[0] == (
        "create_repo",
        ("me/d",),
        {"repo_type": "dataset", "private": False, "exist_ok": True},
    )
    folders = [call for call in api.calls if call[0] == "upload_folder"]
    assert folders[0] == (
        "upload_folder",
        (),
        {
            "repo_id": "me/d",
            "repo_type": "dataset",
            "folder_path": out / "data/website",
            "path_in_repo": "data/website",
            "delete_patterns": "*",
            "commit_message": "Upload data/website",
        },
    )
    assert [c[2]["path_in_repo"] for c in folders] == hub.upload_units(out)
    assert api.calls[-1] == (
        "upload_file",
        (),
        {
            "repo_id": "me/d",
            "repo_type": "dataset",
            "path_or_fileobj": out / "README.md",
            "path_in_repo": "README.md",
            "commit_message": "Update dataset card",
        },
    )
    assert len(api.calls) == 7


def test_publish_private(tmp_path, monkeypatch):
    api = _FakeApi()
    monkeypatch.setattr(hub, "HfApi", lambda: api)
    hub.publish(_out(tmp_path), "me/d", private=True)
    assert api.calls[0][2]["private"] is True


def test_snapshot(tmp_path, monkeypatch):
    seen = {}

    def fake(repo, **kwargs):
        seen.update(kwargs, repo=repo)
        return str(tmp_path)

    monkeypatch.setattr(hub, "snapshot_download", fake)
    assert hub.snapshot("me/d", "v1", tmp_path, ["data/*"]) == tmp_path
    assert seen == {
        "repo": "me/d",
        "repo_type": "dataset",
        "revision": "v1",
        "local_dir": tmp_path,
        "allow_patterns": ["data/*"],
    }


def test_revision_and_list_files(monkeypatch):
    seen = []

    class Info:
        sha = "abc"

    class Api:
        def dataset_info(self, repo):
            seen.append(("info", repo))
            return Info()

        def list_repo_files(self, repo, repo_type, revision):
            seen.append(("files", repo, repo_type, revision))
            return ["a", "b"]

    monkeypatch.setattr(hub, "HfApi", Api)
    assert hub.revision("me/d") == "abc"
    assert hub.list_files("me/d", "abc") == frozenset({"a", "b"})
    assert seen == [("info", "me/d"), ("files", "me/d", "dataset", "abc")]


def test_download(tmp_path, monkeypatch):
    seen = {}

    def fake(repo, path, **kwargs):
        seen.update(kwargs, repo=repo, path=path)
        return str(tmp_path / path)

    monkeypatch.setattr(hub, "hf_hub_download", fake)
    assert hub.download("me/d", "x.parquet", "abc", tmp_path) == Path(tmp_path / "x.parquet")
    assert seen == {
        "repo": "me/d",
        "path": "x.parquet",
        "repo_type": "dataset",
        "revision": "abc",
        "local_dir": tmp_path,
    }
