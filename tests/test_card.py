import polars as pl
from huggingface_hub import DatasetCard

from georeset import card


def test_fmt_and_pct():
    assert card.fmt(1234567) == "1,234,567"
    assert card.pct(0.1234) == "12.3%"


def test_table():
    df = pl.DataFrame({"a": ["x", "y"], "b": [1000, 2]})
    out = card.table(df, {"a": ("Source", str), "b": ("Rows", card.fmt)})
    assert out == "| Source | Rows |\n|---|---:|\n| x | 1,000 |\n| y | 2 |"


def test_configs_cover_sources_languages_and_extras():
    partitions = [("wikipedia", "fra_Latn"), ("website", "fra_Latn"), ("website", "eng_Latn")]
    configs = card.configs(partitions)
    names = [c["config_name"] for c in configs]
    assert names == ["all", "website", "wikipedia", "eng_Latn", "fra_Latn", "unlabelled", "polygons"]
    assert configs[0]["default"] is True
    assert configs[0]["data_files"] == [{"split": "train", "path": "data/*/*/*.parquet"}]
    assert configs[1]["data_files"][0]["path"] == "data/website/*/*.parquet"
    assert configs[4]["data_files"][0]["path"] == "data/*/fra_Latn/*.parquet"
    assert configs[-1]["data_files"][0]["path"] == "polygons/*.parquet"


def test_iso_codes():
    assert card.iso_codes(["fra_Latn", "fra_Cyrl", "und_Zzzz", "eng_Latn"]) == ["eng", "fra"]


def _context():
    by_source = pl.DataFrame(
        {
            "source": ["wikipedia", "total"],
            "sentences": [10, 10],
            "words": [50, 50],
            "polygons": [3, 3],
            "languages": [1, 1],
            "regions": [2, 2],
            "landuse": [0.4, 0.4],
        }
    )
    return card.Context(
        repo_id="me/data",
        by_source=by_source,
        continents=pl.DataFrame({"continent": ["europe"], "sentences": [10], "polygons": [3]}),
        unlabelled=pl.DataFrame({"source": ["wikipedia", "total"], "failed": [1, 1], "unsplit": [2, 2]}),
        partitions=[("wikipedia", "fra_Latn")],
        polygons=7,
        sources={"me/up": "abc1234def"},
        code_url="https://github.com/me/code",
    )


def test_render_is_a_valid_card():
    text = card.render(_context())
    parsed = DatasetCard(text)
    assert parsed.data.license == ["odbl", "cc-by-sa-4.0", "other"]
    assert parsed.data.language == ["fra"]
    assert [c["config_name"] for c in parsed.data.configs][:2] == ["all", "wikipedia"]
    assert "# GeoReSeT Corpus" in text
    assert "| wikipedia | 10 | 50 | 3 | 1 | 2 | 40.0% |" in text
    assert "**total**" in text
    assert "`me/up` @ `abc1234`" in text
    assert 'load_dataset("me/data", "fra_Latn", split="train")' in text
    assert "$" not in text.replace("$ ", "")
