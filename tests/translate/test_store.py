from pathlib import Path
from types import SimpleNamespace

from georeset.translate import store


class _Api:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "list_bucket_tree":
                return [
                    SimpleNamespace(type="file", path="work/b.parquet"),
                    SimpleNamespace(type="directory", path="work/x"),
                    SimpleNamespace(type="file", path="work/a.parquet"),
                ]
            return None

        return call


def test_bucket_calls_the_hub_with_its_id():
    api = _Api()
    bucket = store.Bucket("me/b", api)
    bucket.create()
    assert bucket.files("work/") == ["work/a.parquet", "work/b.parquet"]
    bucket.put([(b"", "claims/x")])
    bucket.delete([])
    bucket.delete(["work/a.parquet"])
    bucket.get([("work/a.parquet", Path("a"))])
    bucket.remove()
    assert api.calls == [
        ("create_bucket", ("me/b",), {"private": True, "exist_ok": True}),
        ("list_bucket_tree", ("me/b",), {"prefix": "work/", "recursive": True}),
        ("batch_bucket_files", ("me/b",), {"add": [(b"", "claims/x")]}),
        ("batch_bucket_files", ("me/b",), {"delete": ["work/a.parquet"]}),
        (
            "download_bucket_files",
            ("me/b", [("work/a.parquet", Path("a"))]),
            {"raise_on_missing_files": True},
        ),
        ("delete_bucket", ("me/b",), {"missing_ok": True}),
    ]


def test_default_bucket_is_private_and_named():
    bucket = store.Bucket(api=_Api())
    assert bucket.id == store.BUCKET == "NoeFlandre/georeset-translation"


def test_default_api_is_the_hub_client(monkeypatch):
    monkeypatch.setattr(store, "HfApi", lambda: "api")
    assert store.Bucket().api == "api"
