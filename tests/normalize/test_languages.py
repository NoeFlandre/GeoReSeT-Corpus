import pytest

from georeset.normalize.languages import UNDETERMINED, normalize


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
        ("fa", "wikimedia", "fas_Arab"),
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


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("als", "gsw_Latn"),
        ("bat-smg", "sgs_Latn"),
        ("be-tarask", "bel_Cyrl"),
        ("bh", "bho_Deva"),
        ("cbk-zam", "cbk_Latn"),
        ("cdo", "cdo_Latn"),
        ("eml", "egl_Latn"),
        ("fiu-vro", "vro_Latn"),
        ("hak", "hak_Latn"),
        ("map-bms", "jav_Latn"),
        ("nan", "nan_Latn"),
        ("nds-nl", "nds_Latn"),
        ("nrm", "nrf_Latn"),
        ("roa-rup", "rup_Latn"),
        ("roa-tara", "nap_Latn"),
        ("sh", "hbs_Latn"),
        ("simple", "eng_Latn"),
        ("zh-classical", "lzh_Hani"),
        ("zh-min-nan", "nan_Latn"),
        ("zh-yue", "yue_Hani"),
    ],
)
def test_wikimedia_special_codes(code, expected):
    assert normalize(code, "wikimedia") == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("az", "azj_Latn"),
        ("ar", "arb_Arab"),
        ("et", "ekk_Latn"),
        ("fa", "fas_Arab"),
        ("ku", "kmr_Latn"),
        ("lv", "lvs_Latn"),
        ("mg", "plt_Latn"),
        ("mn", "khk_Cyrl"),
        ("ms", "zsm_Latn"),
        ("ne", "npi_Deva"),
        ("no", "nob_Latn"),
        ("or", "ory_Orya"),
        ("om", "gaz_Latn"),
        ("ps", "pbt_Arab"),
        ("qu", "quy_Latn"),
        ("sq", "als_Latn"),
        ("sw", "swh_Latn"),
        ("uz", "uzn_Latn"),
        ("yi", "ydd_Hebr"),
        ("zh", "cmn_Hani"),
    ],
)
def test_macrolanguages_map_to_glotlid_individual_language(code, expected):
    assert normalize(code, "wikimedia") == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("cmn", "cmn_Hani"),
        ("gan", "gan_Hani"),
        ("jpn", "jpn_Jpan"),
        ("kor", "kor_Hang"),
        ("lzh", "lzh_Hani"),
        ("wuu", "wuu_Hani"),
        ("yue", "yue_Hani"),
    ],
)
def test_glotlid_script_overrides(code, expected):
    assert normalize(code, "iso639_3") == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [("pbt", "pbt_Arab"), ("khk", "khk_Cyrl"), ("npi", "npi_Deva"), ("ydd", "ydd_Hebr"), ("ory", "ory_Orya")],
)
def test_individual_languages_take_macrolanguage_script(code, expected):
    assert normalize(code, "iso639_3") == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [("ara", "arb_Arab"), ("aze", "azj_Latn"), ("msa", "zsm_Latn"), ("zho", "cmn_Hani"), ("est", "ekk_Latn")],
)
def test_iso639_3_macrolanguages_fold_like_glotlid(code, expected):
    assert normalize(code, "iso639_3") == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("fra_Latn", "French · fra_Latn"),
        ("cmn_Hani", "Mandarin Chinese · cmn_Hani"),
        ("zzz_Zzzz", "zzz_Zzzz"),
    ],
)
def test_display_name(label, expected):
    from georeset.normalize.languages import display_name

    assert display_name(label) == expected
