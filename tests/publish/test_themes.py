import json
from dataclasses import replace

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from georeset.publish import card, geo, themes

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
    assert hexagons.columns == ["hexagon", "n", "pc1", "pc2", "pc3", "l1", "l2", "language"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["sentences"] == n
    assert summary["explained"][0] > summary["explained"][1] > summary["explained"][2] > 0
    ends = pl.read_parquet(out / "ends.parquet")
    assert ends.group_by("pc", "end").len().sort("pc", "end").rows() == [
        (pc, end, 6) for pc in (1, 2, 3, 4, 5) for end in ("high", "low")
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
    # Three global and two local axes, three texts at each end.
    assert text.count("| 3 |") == 10 and "| 4 |" not in text
    assert "# Local axes" in text and "## Local axis 2" in text
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
            "pc": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "end": ["high", "low"] * 5,
            "rank": [1] * 10,
            "id": list("abcdefghij"),
            "text": ["a | b\nc", "x", "y", "z", "u", "v", "p", "q", "r", "s"],
            "language": ["fra_Latn"] * 10,
            "score": [1.0, -1.0] * 5,
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
        local_explained=[0.3, 0.2],
    )
    text = themes.axes_markdown(ends, axes)
    assert "## Local axis 1: airport buildings ↔ lakes and nature reserves" in text
    assert "30.0% of how cells differ from their region." in text
    assert "a \\| b c" in text
    assert "50.0% of the variance; language R² 0.10." in text
    assert text.count("Warning") == 1
    assert "French · fra_Latn 100.0%" in text


def test_compute_with_places_counts_each_text_once_per_hexagon(tmp_path, monkeypatch):
    monkeypatch.setattr(themes, "MIX", 6)
    themes.write_sentences(_data(12), tmp_path / "s", shards=2)
    sentences = sorted((tmp_path / "s").glob("*.parquet"))
    embeddings = _embed(sentences, tmp_path / "e")
    rows = pl.read_parquet(tmp_path / "s" / "*.parquet")
    hexes = geo.h3_parents(
        geo.h3_cells(np.array([lon for _, lon in PLACES]), np.array([lat for lat, _ in PLACES]))
    )
    # Each text has photos at its own place and twice at the first place: texts of place 0 count once
    # in its cell, the others once in their own cell and once in place 0's.
    (lat0, lon0), n = PLACES[0], rows.height
    photos = pl.DataFrame(
        {
            "scene_id": pl.concat([rows["id"]] * 3),
            "lat": [*rows["lat"], *[lat0] * n, *[lat0] * n],
            "lon": [*rows["lon"], *[lon0] * n, *[lon0] * n],
        }
    )
    corpus = replace(themes.LANDUSE, min_texts=10)
    out = themes.compute(sentences, embeddings, tmp_path / "w", corpus, photos)
    hexagons = pl.read_parquet(out / "hexagons.parquet").sort("hexagon")
    counts = dict(zip(hexagons["hexagon"].to_list(), hexagons["n"].to_list(), strict=True))
    assert counts[int(hexes[0])] == 60 and counts[int(hexes[1])] == 12
    summary = json.loads((out / "summary.json").read_text())
    assert summary["sentences"] == 60 and summary["cells"] == 5
    # Axis ends list each text once, wherever it counts.
    ends = pl.read_parquet(out / "ends.parquet")
    assert ends.group_by("pc", "end").agg(pl.col("id").is_unique().all())["id"].all()


def test_pairs_keep_only_one_language_with_its_vectors(tmp_path):
    themes.write_sentences(_data(4), tmp_path / "s", shards=2)
    sentences = sorted((tmp_path / "s").glob("*.parquet"))
    embeddings = _embed(sentences, tmp_path / "e")
    pairs = list(themes._pairs(sentences, embeddings, None, "eng_Latn"))
    rows = pl.concat([rows for rows, _ in pairs])
    assert rows["language"].unique().to_list() == ["eng_Latn"] and rows.height == 4
    for rows, vectors in pairs:
        assert len(vectors) == rows.height
        # Place 3 (English) has the longest vectors, 4 times the unit one plus noise.
        assert np.all(np.linalg.norm(vectors, axis=1) > 3.5)
    assert sum(len(v) for _, v in themes._pairs(sentences, embeddings, None)) == 20


def test_prefetched_keeps_order_whatever_the_lookahead():
    items = list(range(7))
    for ahead in (1, 3, 10):
        assert list(themes._prefetched(lambda x: x * x, items, ahead)) == [x * x for x in items]


def test_cell_means_average_each_cell_then_hexagons_average_cells(tmp_path):
    # Two sibling cells under one display cell: one with 3 texts, one with 1. The display cell is the mean
    # of the two cells, not of the four texts.
    import h3.api.basic_int as h3_int

    parent = geo.h3_parents(geo.h3_cells(np.array([2.35]), np.array([48.85])))[0]
    near = np.array(sorted(h3_int.cell_to_children(int(parent), geo.CELL_RESOLUTION))[:2], np.uint64)
    lon, lat = geo.h3_centres(near)
    rows = pl.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "text": ["1", "2", "3", "4"],
            "language": ["eng_Latn"] * 4,
            "lat": [lat[0]] * 3 + [lat[1]],
            "lon": [lon[0]] * 3 + [lon[1]],
        }
    )
    (tmp_path / "s").mkdir()
    rows.write_parquet(tmp_path / "s" / "x.parquet")
    cells = themes._locate([tmp_path / "s" / "x.parquet"], "", None)
    assert dict(zip(cells.ids.tolist(), cells.texts.tolist(), strict=True)) == {
        int(near[0]): 3,
        int(near[1]): 1,
    }
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    means = themes._cell_means([(rows, vectors)], cells)
    by_cell = dict(zip(cells.ids.tolist(), means.tolist(), strict=True))
    assert by_cell[int(near[0])] == [1.0, 0.0] and by_cell[int(near[1])] == [0.0, 1.0]
    kept, display_means, texts, parents = themes._display(cells, means, 1)
    assert kept.tolist() == [parent] and parents.tolist() == [parent, parent]
    assert display_means.tolist() == [[0.5, 0.5]] and texts.tolist() == [4]
    assert themes._display(cells, means, 5)[0].size == 0
