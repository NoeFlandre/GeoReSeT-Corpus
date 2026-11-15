import polars as pl
import pytest

from georeset.translate import languages


def _nllb(language, text="x"):
    frame = pl.DataFrame(
        {"language": [language], "text": [text]}, schema={"language": pl.String, "text": pl.String}
    )
    return frame.select(languages.nllb(pl.col("language"), pl.col("text"))).item()


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("fra_Latn", "fra_Latn"),
        ("fas_Arab", "pes_Arab"),
        ("ekk_Latn", "est_Latn"),
        ("yue_Hani", "yue_Hant"),
        ("srp_Latn", "srp_Cyrl"),
        ("lat_Latn", None),
        ("fry_Latn", None),
        ("uzn_Cyrl", None),
        ("arb_Latn", None),
        ("und_Zzzz", None),
        (None, None),
    ],
)
def test_nllb_codes(language, expected):
    assert _nllb(language) == expected


def test_chinese_is_split_by_script():
    assert _nllb("cmn_Hani", "这个国家的东部") == "zho_Hans"
    assert _nllb("cmn_Hani", "這個國家的東部") == "zho_Hant"
    assert _nllb("zho_Hani", "這個國家") == "zho_Hant"
    assert _nllb("cmn_Hani", "山水") == "zho_Hans"  # no distinctive character: Simplified
    assert _nllb("cmn_Hani", "這个") == "zho_Hans"  # a tie stays Simplified


def test_script_lists_pair_up_and_do_not_overlap():
    assert len(languages.TRADITIONAL) == len(languages.SIMPLIFIED)
    assert not set(languages.TRADITIONAL) & set(languages.SIMPLIFIED)


def test_nllb_list_holds_the_202_model_tokens():
    assert len(languages.NLLB) == 202
    assert {"eng_Latn", "zho_Hant", "srp_Cyrl", "zul_Latn", "ace_Arab"} <= languages.NLLB


def _model_text(language, text):
    frame = pl.DataFrame({"language": [language], "text": [text]})
    return frame.select(languages.model_text(pl.col("language"), pl.col("text"))).item()


def test_serbian_latin_is_transliterated_digraphs_first():
    assert (
        _model_text("srp_Latn", "Ljubljana i Njegoš, džamija LJ NJ DŽ Dž")
        == "Љубљана и Његош, џамија Љ Њ Џ Џ"
    )
    assert (
        _model_text("srp_Latn", "Đurđevdan, Ćuprija, Čačak, Šabac, Žabalj")
        == "Ђурђевдан, Ћуприја, Чачак, Шабац, Жабаљ"
    )


def test_other_texts_are_unchanged():
    assert _model_text("hrv_Latn", "Ljubljana") == "Ljubljana"
