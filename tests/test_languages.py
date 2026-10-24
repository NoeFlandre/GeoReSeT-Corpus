import pytest

from georeset.languages import UNDETERMINED, normalize


@pytest.mark.parametrize(
    ("code", "scheme", "expected"),
    [
        ("spa_Latn", "glotlid", "spa_Latn"),
        ("zxx_Zxxx", "glotlid", "zxx_Zxxx"),
        ("spa", "glotlid", UNDETERMINED),
        ("fra", "iso639_3", "fra_Latn"),
        ("ita", "iso639_3", "ita_Latn"),
        ("rus", "iso639_3", "rus_Cyrl"),
        ("arb", "iso639_3", "arb_Arab"),
        ("cmn", "iso639_3", "cmn_Hani"),
        ("als", "iso639_3", "als_Latn"),
        ("fr", "wikimedia", "fra_Latn"),
        ("ru", "wikimedia", "rus_Cyrl"),
        ("ar", "wikimedia", "arb_Arab"),
        ("fa", "wikimedia", "pes_Arab"),
        ("zh", "wikimedia", "cmn_Hani"),
        ("ja", "wikimedia", "jpn_Jpan"),
        ("ko", "wikimedia", "kor_Hang"),
        ("sr", "wikimedia", "srp_Cyrl"),
        ("no", "wikimedia", "nob_Latn"),
        ("als", "wikimedia", "gsw_Latn"),
        ("simple", "wikimedia", "eng_Latn"),
        ("zh-yue", "wikimedia", "yue_Hani"),
        ("ceb", "wikimedia", "ceb_Latn"),
    ],
)
def test_normalize(code, scheme, expected):
    assert normalize(code, scheme) == expected


@pytest.mark.parametrize("code", [None, "", "und", "unknown"])
@pytest.mark.parametrize("scheme", ["glotlid", "iso639_3", "wikimedia"])
def test_undetermined(code, scheme):
    assert normalize(code, scheme) == UNDETERMINED


@pytest.mark.parametrize("scheme", ["iso639_3", "wikimedia"])
def test_invalid_code_is_undetermined(scheme):
    assert normalize("not a code!", scheme) == UNDETERMINED


def test_unknown_language_gets_cldr_default_script():
    assert normalize("qaa", "iso639_3") == "qaa_Latn"
