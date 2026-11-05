import json
from dataclasses import replace

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from georeset.publish import card, themes

# Five places, far apart, each with its own direction in a 4-d embedding space.
PLACES = [(48.8, 2.3), (-33.9, 18.4), (35.7, 139.7), (40.7, -74.0), (-22.9, -43.2)]
LANGUAGES = ["fra_Latn", "afr_Latn", "jpn_Jpan", "eng_Latn", "por_Latn"]


def _data(per_place: int) -> pl.LazyFrame:
    rng = np.random.default_rng(0)
    rows = []
    for p, ((lat, lon), language) in enumerate(zip(PLACES, LANGUAGES, strict=True)):
        for i in range(per_place):
            rows.append((f"{p:02d}{i:04d}", f"place {p} sentence {i}", True, language, lat, lon))
    rows.append(("zz", "not relevant", False, "eng_Latn", 0.0, 0.0))
    rows.append(("zy", "place 0 sentence 0", True, "eng_Latn", 1.0, 1.0))  # duplicate text, larger id
    frame = pl.DataFrame(rows, schema=["id", "text", "landuse", "language", "lat", "lon"], orient="row")
    return frame.sample(fraction=1.0, shuffle=True, seed=int(rng.integers(100))).lazy()


def _embed(sentences, out):
    """What scripts/embed.py writes: id, lat, lon and a float16 vector per row."""
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    for path in sentences:
        rows = pq.read_table(path, columns=["id", "lat", "lon", "text"])
        places = [int(t.split()[1]) for t in rows.column("text").to_pylist()]
        vectors = np.eye(5, 4)[places] * (1 + np.arange(5)[places, None]) + rng.normal(
            0, 0.05, (len(places), 4)
        )
        flat = pa.array(vectors.astype(np.float16).reshape(-1), type=pa.float16())
        table = rows.drop_columns("text").append_column("vector", pa.FixedSizeListArray.from_arrays(flat, 4))
        pq.write_table(table, out / path.name)
    return sorted(out.glob("*.parquet"))


@pytest.fixture
def computed(tmp_path, monkeypatch):
    monkeypatch.setattr(themes, "MIX", 6)
    monkeypatch.setattr(themes, "EXTREMES", 3)
    n = themes.write_sentences(_data(12), tmp_path / "sentences", shards=3)
    sentences = sorted((tmp_path / "sentences").glob("*.parquet"))
    embeddings = _embed(sentences, tmp_path / "embeddings")
    corpus = replace(themes.LANDUSE, min_texts=10)
    out = themes.compute(sentences, [str(p) for p in embeddings], tmp_path / "work", corpus)
    return n, out, tmp_path


def test_write_sentences_shards_distinct_relevant_rows(tmp_path):
    n = themes.write_sentences(_data(12), tmp_path, shards=4)
    assert themes.write_sentences(_data(2), tmp_path / "all") == 10
    assert len(list((tmp_path / "all").glob("*.parquet"))) == themes.LANDUSE.shards
    assert n == 60
    names = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert names == [themes.shard_name(i, 4) for i in range(4)]
    assert names[0] == "train-00000-of-00004.parquet"
    rows = pl.read_parquet(tmp_path / "*.parquet")
    assert rows.columns == ["id", "text", "language", "lat", "lon"]
    assert rows["id"].to_list() == sorted(rows["id"].to_list())
    assert rows["text"].n_unique() == 60
    assert "zy" not in rows["id"]


def test_read_vectors(tmp_path):
    themes.write_sentences(_data(2), tmp_path / "s", shards=1)
    (path,) = _embed(sorted((tmp_path / "s").glob("*.parquet")), tmp_path / "e")
    ids, vectors = themes.read_vectors(str(path))
    assert ids.to_list() == pl.read_parquet(path)["id"].to_list()
    assert vectors.dtype == np.float32 and vectors.shape == (10, 4)


def test_write_sentences_of_scenes(tmp_path):
    scenes = pl.LazyFrame(
        {"scene_id": [2, 1, 3], "text": ["b", "a", "a"], "lat": [1.0, 2.0, 3.0], "lon": [0.0, 0.0, 0.0]}
    )
    assert themes.write_sentences(scenes, tmp_path, themes.SCENES) == 2
    (path,) = tmp_path.glob("*.parquet")
    assert path.name == "train-00000-of-00001.parquet"
    assert pl.read_parquet(path).rows() == [(1, "a", "eng_Latn", 2.0, 0.0), (2, "b", "eng_Latn", 1.0, 0.0)]


def test_compute(computed):
    n, out, _ = computed
    assert out.name == "landuse"
    hexagons = pl.read_parquet(out / "hexagons.parquet")
    assert hexagons.height == 5
    assert hexagons["n"].to_list() == [12] * 5
    assert sorted(hexagons["language"].to_list()) == sorted(LANGUAGES)
    assert hexagons.columns == ["hexagon", "n", "pc1", "pc2", "pc3", "language"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["sentences"] == n
    assert summary["explained"][0] > summary["explained"][1] > summary["explained"][2] > 0
    ends = pl.read_parquet(out / "ends.parquet")
    assert ends.group_by("pc", "end").len().sort("pc", "end").rows() == [
        (pc, end, 6) for pc in (1, 2, 3) for end in ("high", "low")
    ]
    high = ends.filter(pc=1, end="high")
    assert high["rank"].to_list() == list(range(1, 7))
    assert high["score"].is_sorted(descending=True)
    assert ends.filter(pc=1, end="low")["score"].is_sorted()
    # The place with the longest vectors (place 3) sits at one end of the first axis.
    assert ends.filter(pc=1)["text"].str.starts_with("place 3").any()


def test_compute_rejects_mismatched_shards(tmp_path):
    themes.write_sentences(_data(2), tmp_path / "s", shards=2)
    sentences = sorted((tmp_path / "s").glob("*.parquet"))
    embeddings = _embed(sentences, tmp_path / "e")
    with pytest.raises(ValueError, match="row for row"):
        themes.compute(sentences, embeddings[::-1], tmp_path / "w")


def test_render(computed, monkeypatch):
    _, _, tmp_path = computed
    monkeypatch.setattr(themes, "EXTREMES", 3)
    corpus = replace(themes.LANDUSE, min_texts=10, axes={1: ("towns", "farms"), 2: ("", ""), 3: ("", "")})
    result = themes.render(tmp_path / "work", tmp_path / "assets", corpus)
    report = tmp_path / "work" / "pca_axes.md"
    assert isinstance(result, card.Themes)
    assert result.hexagons == 5 and result.corpus.min_texts == 10
    assert len(result.language_r2) == 3
    # One language per hexagon: language explains every axis.
    assert result.language_r2 == pytest.approx([1.0, 1.0, 1.0])
    for name in ("map_pca.png", "map_pc1.png", "map_pc2.png", "map_pc3.png"):
        assert (tmp_path / "assets" / name).read_bytes()[:4] == b"\x89PNG"
    text = report.read_text()
    assert "## PC1: towns ↔ farms" in text
    assert "### Highest (farms)" in text and "### Lowest (towns)" in text
    assert "## PC2\n" in text and "### Highest\n" in text
    assert text.count("**Warning: mostly language.**") == 3
    assert text.count("| 3 |") == 6 and "| 4 |" not in text
    assert "Languages of the 6 sentences at this end:" in text


def test_render_scenes_names_files_and_skips_languages(computed):
    _, out, tmp_path = computed
    work = tmp_path / "work"
    out.rename(work / "themes" / "scenes")
    corpus = replace(themes.SCENES, min_texts=10)
    result = themes.render(work, tmp_path / "assets", corpus)
    assert result.corpus is corpus
    for name in ("map_pca_scenes.png", "map_pc1_scenes.png", "map_pc3_scenes.png"):
        assert (tmp_path / "assets" / name).is_file()
    text = (work / "pca_axes_scenes.md").read_text()
    assert text.startswith("# What places talk about: the PCA axes of the scene descriptions")
    assert "with 10 descriptions or more" in text
    assert "language" not in text.lower()
    assert "| # | Score | Text |" in text


def test_axes_markdown_escapes_and_quiet_axes():
    ends = pl.DataFrame(
        {
            "pc": [1, 1, 2, 2, 3, 3],
            "end": ["high", "low"] * 3,
            "rank": [1] * 6,
            "id": list("abcdef"),
            "text": ["a | b\nc", "x", "y", "z", "u", "v"],
            "language": ["fra_Latn"] * 6,
            "score": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        },
        schema_overrides={"rank": pl.UInt32},
    )
    axes = card.Themes(
        corpus=replace(themes.LANDUSE, min_texts=1),
        model="m",
        prompt="p",
        texts=6,
        hexagons=3,
        explained=[0.5, 0.2, 0.1],
        language_r2=[0.1, 0.6, 0.5],
        warning=0.5,
    )
    text = themes.axes_markdown(ends, axes)
    assert "a \\| b c" in text
    assert "50.0% of the variance; language R² 0.10." in text
    assert text.count("Warning") == 1
    assert "French · fra_Latn 100.0%" in text
