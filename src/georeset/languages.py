"""Normalize language codes from all sources to GlotLID labels (``fra_Latn``)."""

import re
from functools import cache
from typing import Literal

import langcodes

Scheme = Literal["glotlid", "iso639_3", "wikimedia"]
UNDETERMINED = "und_Zzzz"
_LABEL = re.compile(r"[a-z]{3}_[A-Z][a-z]{3}")

# Wikimedia edition codes that are not ISO codes, or mean something else in ISO.
_WIKIMEDIA: dict[str, str] = {
    "als": "gsw_Latn",
    "bat-smg": "sgs_Latn",
    "be-tarask": "bel_Cyrl",
    "bh": "bho_Deva",
    "cbk-zam": "cbk_Latn",
    "cdo": "cdo_Latn",
    "eml": "egl_Latn",
    "fiu-vro": "vro_Latn",
    "hak": "hak_Latn",
    "map-bms": "jav_Latn",
    "nan": "nan_Latn",
    "nds-nl": "nds_Latn",
    "nrm": "nrf_Latn",
    "roa-rup": "rup_Latn",
    "roa-tara": "nap_Latn",
    "sh": "hbs_Latn",
    "simple": "eng_Latn",
    "zh-classical": "lzh_Hani",
    "zh-min-nan": "nan_Latn",
    "zh-yue": "yue_Hani",
}

# ISO macrolanguages to the individual language GlotLID predicts.
_MACRO: dict[str, str] = {
    "aze": "azj",
    "ara": "arb",
    "est": "ekk",
    "fas": "pes",
    "kur": "kmr",
    "lav": "lvs",
    "mlg": "plt",
    "mon": "khk",
    "msa": "zsm",
    "nep": "npi",
    "nor": "nob",
    "ori": "ory",
    "orm": "gaz",
    "pus": "pbt",
    "que": "quy",
    "sqi": "als",
    "swa": "swh",
    "uzb": "uzn",
    "yid": "ydd",
    "zho": "cmn",
}

# Scripts GlotLID uses where CLDR's likely script differs.
_SCRIPT: dict[str, str] = {
    "cmn": "Hani",
    "gan": "Hani",
    "jpn": "Jpan",
    "kor": "Hang",
    "lzh": "Hani",
    "wuu": "Hani",
    "yue": "Hani",
}


def _script(iso3: str) -> str:
    """GlotLID script override, else CLDR's likely script (``Latn`` when unknown)."""
    return _SCRIPT.get(iso3) or str(langcodes.Language.get(iso3).maximize().script)


def _label(iso3: str) -> str:
    return f"{iso3}_{_script(iso3)}"


def _wikimedia(code: str) -> str:
    if code in _WIKIMEDIA:
        return _WIKIMEDIA[code]
    iso3 = langcodes.Language.get(code).to_alpha3()
    return _label(_MACRO.get(iso3, iso3))


def _glotlid(code: str) -> str:
    if _LABEL.fullmatch(code):
        return code
    raise ValueError(code)


_CONVERTERS = {"glotlid": _glotlid, "iso639_3": _label, "wikimedia": _wikimedia}


@cache
def normalize(code: str | None, scheme: Scheme) -> str:
    """Return the GlotLID label for ``code`` written in ``scheme``, or ``und_Zzzz``."""
    if not code or code in ("und", "unknown"):
        return UNDETERMINED
    try:
        return _CONVERTERS[scheme](code)
    except (LookupError, ValueError):
        return UNDETERMINED
