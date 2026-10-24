from georeset import hub


class _FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create", repo_id, kwargs["private"]))
        return f"https://hf.co/datasets/{repo_id}"

    def upload_folder(self, **kwargs):
        self.calls.append(("folder", kwargs["path_in_repo"]))

    def upload_file(self, **kwargs):
        self.calls.append(("file", kwargs["path_in_repo"]))


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


def test_publish_is_public_and_card_goes_last(tmp_path, monkeypatch):
    api = _FakeApi()
    monkeypatch.setattr(hub, "HfApi", lambda: api)
    assert hub.publish(_out(tmp_path), "me/d") == "https://hf.co/datasets/me/d"
    assert api.calls[0] == ("create", "me/d", False)
    assert api.calls[-1] == ("file", "README.md")
    assert len(api.calls) == 7


def test_revision_and_list_files(monkeypatch):
    class Info:
        sha = "abc"

    class Api:
        def dataset_info(self, repo):
            return Info()

        def list_repo_files(self, repo, repo_type, revision):
            assert (repo_type, revision) == ("dataset", "abc")
            return ["a", "b"]

    monkeypatch.setattr(hub, "HfApi", Api)
    assert hub.revision("me/d") == "abc"
    assert hub.list_files("me/d", "abc") == frozenset({"a", "b"})


def test_download(tmp_path, monkeypatch):
    seen = {}

    def fake(repo, path, **kwargs):
        seen.update(kwargs, repo=repo, path=path)
        return str(tmp_path / path)

    monkeypatch.setattr(hub, "hf_hub_download", fake)
    assert hub.download("me/d", "x.parquet", "abc", tmp_path) == tmp_path / "x.parquet"
    assert seen["revision"] == "abc"
    assert seen["repo_type"] == "dataset"
    assert seen["local_dir"] == tmp_path
