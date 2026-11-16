import hashlib

import numpy as np
import polars as pl

from georeset.translate import route

SCHEMA = {"id": pl.String, "text": pl.String, "language": pl.String, "source": pl.String, "status": pl.String}


def _rows(*rows):
    return pl.DataFrame([dict(zip(route.COLUMNS, row, strict=True)) for row in rows], schema=SCHEMA)


def test_overrides_keep_confident_labels_that_differ_from_the_page():
    rows = _rows(
        ("a", "Home", "hin_Deva", "website", None),
        ("b", "Bonjour", "hin_Deva", "website", None),
        ("c", "नमस्ते", "hin_Deva", "website", None),
    )
    out = route.overrides(rows, ["eng_Latn", "fra_Latn", "hin_Deva"], [route.THRESHOLD, 0.89, 1.0])
    assert out.to_dicts() == [{"id": "a", "detected": "eng_Latn"}]


def test_key_is_the_sha256_prefix_of_code_and_text():
    assert route.key("fra_Latn", "Bonjour") == hashlib.sha256(b"fra_Latn\nBonjour").digest()[:16]
    assert route.key("fra_Latn", "Bonjour") != route.key("ita_Latn", "Bonjour")


def test_routed_rows():
    rows = _rows(
        ("1", "Bonjour", "fra_Latn", "wikipedia", None),
        ("2", "Hello", "eng_Latn", "website", None),
        ("3", "Home", "hin_Deva", "website", None),
        ("4", "Salve", "lat_Latn", "wikipedia", None),
        ("5", "Ganze Seite", "deu_Latn", "website", "unsplit"),
        ("6", "Whole page", "eng_Latn", "website", "unsplit"),
        ("7", "Hallo", "eng_Latn", "website", None),
        ("8", "Beograd", "srp_Latn", "osm_description", "failed"),
    )
    detected = pl.DataFrame({"id": ["3", "7"], "detected": ["eng_Latn", "deu_Latn"]})
    out = route.routed(rows.lazy(), detected.lazy()).collect()
    assert out.columns == [*route.COLUMNS, "translation", "model_text", "translation_language", "key"]
    assert out["id"].to_list() == [str(i) for i in range(1, 9)]
    assert out.select("translation", "translation_language").rows() == [
        ("nllb", "fra_Latn"),
        ("original", None),
        ("original", None),
        ("unsupported", None),
        ("unsplit", None),
        ("original", None),
        ("nllb", "deu_Latn"),
        ("nllb", "srp_Cyrl"),
    ]
    assert out["model_text"][7] == "Београд"
    assert out["key"].to_list() == [
        route.key("fra_Latn", "Bonjour"),
        None,
        None,
        None,
        None,
        None,
        route.key("deu_Latn", "Hallo"),
        route.key("srp_Cyrl", "Београд"),
    ]


def test_overrides_compare_probabilities_in_float64():
    rows = _rows(("a", "Home", "hin_Deva", "website", None))
    probability = np.array([route.THRESHOLD], dtype=np.float32)  # 0.89999998 once widened
    assert route.overrides(rows, ["eng_Latn"], probability).height == 0


def test_overrides_of_no_rows_keep_their_schema():
    out = route.overrides(_rows(), [], [])
    assert out.schema == {"id": pl.String, "detected": pl.String}


def test_routed_rows_keep_their_order():
    rows = _rows(*[(str(i), "Home", "hin_Deva", "website", None) for i in range(300)])
    detected = pl.DataFrame(
        {"id": [str(i) for i in reversed(range(0, 300, 3))], "detected": ["eng_Latn"] * 100}
    )
    out = route.routed(rows.lazy(), detected.lazy()).collect()
    assert out["id"].to_list() == [str(i) for i in range(300)]
    assert out["translation"].to_list() == ["original" if i % 3 == 0 else "nllb" for i in range(300)]
