import polars as pl
import pytest

from georeset.pipeline import Layout, build_meta, build_rows, build_shards
from georeset.sources import Staged
from georeset.stage import write_staged


def _polygons(ids, name):
    n = len(ids)
    return pl.DataFrame(
        {
            "polygon_id": ids,
            "name": [name] * n,
            "osm_tag": ["landuse=forest"] * n,
            "lat": [1.0] * n,
            "lon": [2.0] * n,
            "area_m2": [3.0] * n,
            "geometry": [b"\x01"] * n,
        }
    )


def _sentences(source, doc, labels, language="fra_Latn"):
    n = len(labels)
    return pl.DataFrame(
        {
            "source": [source] * n,
            "doc_key": [doc] * n,
            "position": list(range(n)),
            "text": [f"{doc} sentence {i}" for i in range(n)],
            "label": labels,
            "failure_reason": [None] * n,
            "language": [language] * n,
            "url": ["u"] * n,
        },
        schema_overrides={"position": pl.Int32},
    )


@pytest.fixture
def layout(tmp_path):
    layout = Layout(tmp_path / "work", tmp_path / "out")
    # The same wiki doc is staged in two overlapping regions; centro is the smaller extract.
    for stem, ids in [
        ("italy-latest", ["way/1", "way/2", "way/3"]),
        ("centro-latest", ["way/1"]),
    ]:
        write_staged(
            Staged(
                polygons=_polygons(ids, stem),
                links=pl.DataFrame({"doc_key": ["Q1"], "polygon_id": ["way/1"]}),
                sentences=_sentences("wikipedia", "Q1", ["yes", "no", "failed"]),
            ),
            layout.stage / "wiki" / stem,
        )
    write_staged(
        Staged(
            polygons=_polygons(["way/2"], "web"),
            links=pl.DataFrame({"doc_key": ["http://x"], "polygon_id": ["way/2"]}),
            sentences=_sentences("website", "http://x", ["no", "yes"], "eng_Latn"),
        ),
        layout.stage / "website" / "italy-latest",
    )
    return layout


def _build(layout):
    build_meta(layout)
    for staged in sorted(layout.stage.glob("*/*")):
        build_rows(layout, staged)
    build_shards(layout, workers=2)


def test_end_to_end(layout):
    _build(layout)
    out = layout.out
    assert sorted(str(p.relative_to(out)) for p in out.rglob("*.parquet")) == [
        "data/website/eng_Latn/train-00000-of-00001.parquet",
        "data/wikipedia/fra_Latn/train-00000-of-00001.parquet",
        "polygons/train-00000-of-00001.parquet",
        "unlabelled/train-00000-of-00001.parquet",
    ]
    wiki = pl.read_parquet(out / "data/wikipedia")
    assert wiki["landuse"].to_list() == [True, False]
    assert wiki["region"].to_list() == ["centro", "centro"]
    assert wiki["polygon_ids"].to_list() == [["way/1"], ["way/1"]]
    assert pl.read_parquet(out / "unlabelled").height == 1
    polygons = pl.read_parquet(out / "polygons").sort("polygon_id")
    assert polygons["region"].to_list() == ["centro", "italy", "italy"]
    assert polygons["name"].to_list() == [
        "centro-latest",
        "italy-latest",
        "italy-latest",
    ]


def test_rerun_is_a_no_op_and_work_is_cleaned(layout):
    _build(layout)
    first = pl.read_parquet(layout.out / "data/wikipedia")
    _build(layout)
    assert pl.read_parquet(layout.out / "data/wikipedia").equals(first)
    assert list(layout.stage.glob("*/*")) == []
    assert list((layout.work / "rows").glob("data/*/*")) == []
    assert not (layout.work / "rows" / "unlabelled").exists()


def test_rows_skip_region_without_docs(layout):
    build_meta(layout)
    (layout.work / "meta" / "docs" / "italy-latest.parquet").unlink()
    build_rows(layout, layout.stage / "website" / "italy-latest")
    assert not (layout.work / "rows").exists()
