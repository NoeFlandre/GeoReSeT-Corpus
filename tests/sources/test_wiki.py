import polars as pl
import shapely
from conftest import SQUARE

from georeset.sources import wiki


def _wiki_project(write, project, rows):
    sentences = write(
        f"{project}_sentences",
        {
            "sentence_id": [r[0] for r in rows],
            "document_id": [r[1] for r in rows],
            "section_index": [r[2] for r in rows],
            "sentence_index": [r[3] for r in rows],
            "text": [r[4] for r in rows],
        },
    )
    labels = write(
        f"{project}_labels",
        {
            "sentence_id": [r[0] for r in rows],
            "decision": ["yes"] * len(rows),
            "failure_reason": [None] * len(rows),
            "language": [r[5] for r in rows],
        },
        schema={k: pl.String for k in ["sentence_id", "decision", "failure_reason", "language"]},
    )
    return {f"{project}_sentences": sentences, f"{project}_labels": labels}


def test_wiki_transform_orders_sections_and_builds_permalinks(write):
    paths = {
        "polygons": write(
            "polygons",
            {
                "osm_type": ["relation"],
                "osm_id": [9],
                "name": ["Forest"],
                "tags": ['{"landuse":"forest"}'],
                "lat": [1.0],
                "lon": [2.0],
                "area_m2": [3.0],
                "geometry": [SQUARE],
            },
        ),
        "links": write(
            "links",
            {
                "osm_type": ["relation"],
                "osm_id": [9],
                "document_id": ["Q1:wikipedia:fr:10:99"],
            },
        ),
    }
    doc = "Q1:wikipedia:fr:10:99"
    paths |= _wiki_project(
        write,
        "wikipedia",
        [
            ("c", doc, 1, 1, "Third part", "fr"),
            ("e", doc, 1, 0, "[ modifier | modifier le code ]", "fr"),
            ("b", doc, 0, 1, "Second part", "fr"),
            ("a", doc, 0, 0, "First part", "fr"),
            ("d", doc, 0, 2, "[ edit ] [1]", "fr"),
        ],
    )
    staged = wiki.transform(paths)
    s = staged.sentences.sort("position")
    assert s["text"].to_list() == ["First part", "Second part", "Third part"]
    assert s["position"].to_list() == [0, 1, 2]
    assert set(s["url"]) == {"https://fr.wikipedia.org/w/index.php?oldid=99"}
    assert set(s["source"]) == {"wikipedia"}
    assert set(s["language"]) == {"fra_Latn"}
    assert set(s["label"]) == {"yes"}
    assert staged.links.rows() == [(doc, "relation/9")]
    assert shapely.from_wkb(staged.polygons["geometry"].item()).area == 1.0
    assert staged.polygons.drop("geometry").rows() == [
        ("relation/9", "Forest", "landuse=forest", 1.0, 2.0, 3.0)
    ]


def test_wiki_files_lists_both_projects():
    assert wiki.files("x") == {
        "polygons": "polygons/x.parquet",
        "links": "polygon_document_links/x.parquet",
        "wikipedia_sentences": "wikipedia/sentences/x.parquet",
        "wikipedia_labels": "labels/wikipedia/sentences/x.parquet",
        "wikivoyage_sentences": "wikivoyage/sentences/x.parquet",
        "wikivoyage_labels": "labels/wikivoyage/sentences/x.parquet",
    }


def test_bracket_only_rows_are_dropped_but_bracketed_prose_is_kept():
    import polars as pl

    texts = [
        "[ edit | edit source ]",
        "[ editar código · editar ]",
        "[Wadel 2005]",
        "[ edit ] [2]",
        "[ 編集 | ソースを編集 ] 10-17h",
        "Built in 1805 [3].",
        "A [small] lake.",
    ]
    kept = pl.DataFrame({"t": texts}).filter(~pl.col("t").str.contains(wiki.BRACKETS_ONLY))["t"].to_list()
    assert kept == ["[ 編集 | ソースを編集 ] 10-17h", "Built in 1805 [3].", "A [small] lake."]


def test_permalink():
    out = pl.select(wiki.permalink(pl.lit("Q2:wikivoyage:en:5:42"))).item()
    assert out == "https://en.wikivoyage.org/w/index.php?oldid=42"
