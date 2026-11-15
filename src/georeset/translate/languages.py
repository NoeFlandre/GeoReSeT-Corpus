"""Corpus language labels (GlotLID) to the NLLB-200 source codes, as polars expressions.

Most labels are NLLB codes already. Some name the same language differently (``fas_Arab`` is ``pes_Arab``),
Chinese is split by script (``cmn_Hani`` is either ``zho_Hans`` or ``zho_Hant``), and Serbian in Latin script
is transliterated to Cyrillic, which NLLB knows. Everything else outside NLLB is unsupported (null).
"""

import polars as pl

ENGLISH = "eng_Latn"

# The 202 language tokens of facebook/nllb-200-3.3B (special_tokens_map.json).
NLLB = frozenset(
    """
    ace_Arab ace_Latn acm_Arab acq_Arab aeb_Arab afr_Latn ajp_Arab aka_Latn amh_Ethi apc_Arab arb_Arab
    ars_Arab ary_Arab arz_Arab asm_Beng ast_Latn awa_Deva ayr_Latn azb_Arab azj_Latn bak_Cyrl bam_Latn
    ban_Latn bel_Cyrl bem_Latn ben_Beng bho_Deva bjn_Arab bjn_Latn bod_Tibt bos_Latn bug_Latn bul_Cyrl
    cat_Latn ceb_Latn ces_Latn cjk_Latn ckb_Arab crh_Latn cym_Latn dan_Latn deu_Latn dik_Latn dyu_Latn
    dzo_Tibt ell_Grek eng_Latn epo_Latn est_Latn eus_Latn ewe_Latn fao_Latn pes_Arab fij_Latn fin_Latn
    fon_Latn fra_Latn fur_Latn fuv_Latn gla_Latn gle_Latn glg_Latn grn_Latn guj_Gujr hat_Latn hau_Latn
    heb_Hebr hin_Deva hne_Deva hrv_Latn hun_Latn hye_Armn ibo_Latn ilo_Latn ind_Latn isl_Latn ita_Latn
    jav_Latn jpn_Jpan kab_Latn kac_Latn kam_Latn kan_Knda kas_Arab kas_Deva kat_Geor knc_Arab knc_Latn
    kaz_Cyrl kbp_Latn kea_Latn khm_Khmr kik_Latn kin_Latn kir_Cyrl kmb_Latn kon_Latn kor_Hang kmr_Latn
    lao_Laoo lvs_Latn lij_Latn lim_Latn lin_Latn lit_Latn lmo_Latn ltg_Latn ltz_Latn lua_Latn lug_Latn
    luo_Latn lus_Latn mag_Deva mai_Deva mal_Mlym mar_Deva min_Latn mkd_Cyrl plt_Latn mlt_Latn mni_Beng
    khk_Cyrl mos_Latn mri_Latn zsm_Latn mya_Mymr nld_Latn nno_Latn nob_Latn npi_Deva nso_Latn nus_Latn
    nya_Latn oci_Latn gaz_Latn ory_Orya pag_Latn pan_Guru pap_Latn pol_Latn por_Latn prs_Arab pbt_Arab
    quy_Latn ron_Latn run_Latn rus_Cyrl sag_Latn san_Deva sat_Beng scn_Latn shn_Mymr sin_Sinh slk_Latn
    slv_Latn smo_Latn sna_Latn snd_Arab som_Latn sot_Latn spa_Latn als_Latn srd_Latn srp_Cyrl ssw_Latn
    sun_Latn swe_Latn swh_Latn szl_Latn tam_Taml tat_Cyrl tel_Telu tgk_Cyrl tgl_Latn tha_Thai tir_Ethi
    taq_Latn taq_Tfng tpi_Latn tsn_Latn tso_Latn tuk_Latn tum_Latn tur_Latn twi_Latn tzm_Tfng uig_Arab
    ukr_Cyrl umb_Latn urd_Arab uzn_Latn vec_Latn vie_Latn war_Latn wol_Latn xho_Latn ydd_Hebr yor_Latn
    yue_Hant zho_Hans zho_Hant zul_Latn
    """.split()  # noqa: SIM905 (a block of codes reads better than a 202-item list)
)

# Corpus labels naming an NLLB language under another code.
ALIASES = {
    "fas_Arab": "pes_Arab",
    "ekk_Latn": "est_Latn",
    "yue_Hani": "yue_Hant",
    "srp_Latn": "srp_Cyrl",  # after transliteration, see ``model_text``
}
CHINESE = ("cmn_Hani", "zho_Hani")
SERBIAN_LATIN = "srp_Latn"

# Frequent characters written differently in Traditional and Simplified Chinese (same order in both).
TRADITIONAL = "們個這說國學會來時開對發東長門問間關書車馬見現進還過經從動實點兩處區號華灣廣歷為與業電機線後頭"
SIMPLIFIED = "们个这说国学会来时开对发东长门问间关书车马见现进还过经从动实点两处区号华湾广历为与业电机线后头"

# Serbian Latin to Cyrillic; digraphs first, so that ``lj`` is read before ``l``.
_DIGRAPHS = {
    "Lj": "Љ",
    "LJ": "Љ",
    "lj": "љ",
    "Nj": "Њ",
    "NJ": "Њ",
    "nj": "њ",
    "Dž": "Џ",
    "DŽ": "Џ",
    "dž": "џ",
}
_LETTERS = dict(
    zip(
        "ABVGDĐEŽZIJKLMNOPRSTĆUFHCČŠabvgdđežzijklmnoprstćufhcčš",
        "АБВГДЂЕЖЗИЈКЛМНОПРСТЋУФХЦЧШабвгдђежзијклмнопрстћуфхцчш",
        strict=True,
    )
)
SERBIAN = {**_DIGRAPHS, **_LETTERS}


def _count(text: pl.Expr, characters: str) -> pl.Expr:
    return text.str.count_matches(f"[{characters}]")


def chinese(text: pl.Expr) -> pl.Expr:
    """``zho_Hant`` when a text has more Traditional-only than Simplified-only characters, else Simplified."""
    traditional = _count(text, TRADITIONAL) > _count(text, SIMPLIFIED)
    return pl.when(traditional).then(pl.lit("zho_Hant")).otherwise(pl.lit("zho_Hans"))


def nllb(language: pl.Expr, text: pl.Expr) -> pl.Expr:
    """The NLLB source code for a row, or null when NLLB does not know its language."""
    code = pl.when(language.is_in(CHINESE)).then(chinese(text)).otherwise(language.replace(ALIASES))
    return pl.when(code.is_in(list(NLLB))).then(code)


def to_cyrillic(text: pl.Expr) -> pl.Expr:
    return text.str.replace_many(SERBIAN, leftmost=True)


def model_text(language: pl.Expr, text: pl.Expr) -> pl.Expr:
    """The text given to the model: Serbian Latin in Cyrillic, everything else unchanged."""
    return pl.when(language == SERBIAN_LATIN).then(to_cyrillic(text)).otherwise(text)
