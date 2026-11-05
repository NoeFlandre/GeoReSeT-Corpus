import os
from pathlib import Path

import polars as pl
import pytest
from huggingface_hub import DatasetCard

from georeset.publish import card


def test_fmt_and_pct():
    assert card.fmt(1234567) == "1,234,567"
    assert card.pct(0.1234) == "12.3%"


def test_table():
    df = pl.DataFrame({"a": ["x", "y"], "b": [1000, 2]})
    out = card.table(df, {"a": ("Source", str), "b": ("Rows", card.fmt)})
    assert out == "| Source | Rows |\n|---|---:|\n| x | 1,000 |\n| y | 2 |"


def test_configs_cover_sources_languages_and_extras():
    partitions = [("wikipedia", "fra_Latn"), ("website", "fra_Latn"), ("website", "eng_Latn")]
    configs = card.configs(partitions)
    names = [c["config_name"] for c in configs]
    assert names == ["all", "website", "wikipedia", "eng_Latn", "fra_Latn", "unlabelled", "polygons"]
    assert configs[0]["default"] is True
    assert configs[0]["data_files"] == [{"split": "train", "path": "data/*/*/*.parquet"}]
    assert configs[1]["data_files"][0]["path"] == "data/website/*/*.parquet"
    assert configs[4]["data_files"][0]["path"] == "data/*/fra_Latn/*.parquet"
    assert configs[-1]["data_files"][0]["path"] == "polygons/*.parquet"


def test_folder_configs():
    assert card.folder_configs(card.SCENE_CONFIGS) == [
        {"config_name": "scenes", "data_files": [{"split": "train", "path": "scenes/*.parquet"}]},
        {"config_name": "scene_photos", "data_files": [{"split": "train", "path": "scene_photos/*.parquet"}]},
    ]
    assert card.folder_configs(()) == []


@pytest.mark.parametrize(
    ("n", "label"),
    [
        (0, "n<1K"),
        (999, "n<1K"),
        (1000, "1K<n<10K"),
        (99_999_999, "10M<n<100M"),
        (10**8, "100M<n<1B"),
        (10**9, "1B<n<10B"),
    ],
)
def test_size_category(n, label):
    assert card.size_category(n) == label


def test_iso_codes():
    assert card.iso_codes(["fra_Latn", "fra_Cyrl", "und_Zzzz", "eng_Latn"]) == ["eng", "fra"]


def _context():
    by_source = pl.DataFrame(
        {
            "source": ["wikipedia", "total"],
            "sentences": [10, 10],
            "words": [50, 50],
            "polygons": [3, 3],
            "languages": [1, 1],
            "regions": [2, 2],
            "landuse": [0.4, 0.4],
        }
    )
    return card.Context(
        repo_id="me/data",
        by_source=by_source,
        continents=pl.DataFrame({"continent": ["europe"], "sentences": [10], "polygons": [3]}),
        unlabelled=pl.DataFrame({"source": ["wikipedia", "total"], "failed": [1, 1], "unsplit": [2, 2]}),
        partitions=[("wikipedia", "fra_Latn")],
        polygons=7,
        areas=pl.DataFrame({"p10": [10.0], "p25": [25.0], "p50": [500.0], "p75": [20_000.0], "p90": [3e6]}),
        sources={"me/up": "abc1234def"},
        code_url="https://github.com/me/code",
    )


def _scenes():
    return card.Scenes(
        totals={"scenes": 2, "photos": 1500, "median_photos": 750, "words": 40, "countries": 3},
        labels=pl.DataFrame(
            {"lulc_label": ["Tundra", "Lake / Pond"], "scenes": [1, 1], "photos": [1000, 500]}
        ),
        platforms=pl.DataFrame({"platform": ["flickr"], "photos": [1500], "linked": [0.25]}),
        locality=pl.DataFrame({"km10": [0], "km100": [1], "km1000": [2]}),
        raw={"me/raw": "f" * 40},
    )


def _fused():
    return card.Context(**{**_context().__dict__, "scenes": _scenes()})


def test_render_is_a_valid_card():
    text = card.render(_context())
    parsed = DatasetCard(text)
    assert parsed.data.license == ["odbl", "cc-by-sa-4.0", "other"]
    assert parsed.data.language == ["fra"]
    assert parsed.data.size_categories == ["n<1K"]
    assert [c["config_name"] for c in parsed.data.configs][:2] == ["all", "wikipedia"]
    assert "# GeoReSeT Corpus" in text
    assert "| wikipedia | 10 | 50 | 3 | 1 | 2 | 40.0% |" in text
    assert "**total**" in text
    assert "`me/up` @ `abc1234`" in text
    assert 'load_dataset("me/data", "fra_Latn", split="train")' in text
    assert "$" not in text.replace("$ ", "")


SNAPSHOT = Path(__file__).parents[1] / "data" / "card_snapshot.md"


def test_render_matches_snapshot():
    """Every heading, label and table of the card.

    Regenerate with ``UPDATE_SNAPSHOTS=1`` after reviewing a change.
    """
    text = card.render(_context())
    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOT.write_text(text)
    assert text == SNAPSHOT.read_text()


def test_fused_card_is_valid_and_mentions_scenes():
    text = card.render(_fused())
    parsed = DatasetCard(text)
    assert parsed.data.pretty_name == "GeoReSeT Corpus Fused"
    assert parsed.data.license == ["odbl", "cc-by-sa-4.0", "cc-by-4.0", "other"]
    assert parsed.data.tags[-2:] == ["street-level-imagery", "scene-description"]
    assert [c["config_name"] for c in parsed.data.configs][-2:] == ["scenes", "scene_photos"]
    assert "# GeoReSeT Corpus Fused\n" in text
    assert "**Plus 2 English scene descriptions" in text
    assert "| Tundra | 1 | 1,000 |" in text
    assert "| flickr | 1,500 | 25.0% |" in text
    assert "| 10 km | 100 km | 1,000 km |" in text
    assert "`me/raw` @ `fffffff`" in text
    assert "`polygons`, `scenes`, `scene_photos`." in text
    assert "7. **Scenes**" in text
    assert 'load_dataset("me/data", "scene_photos", split="train")' in text
    assert "$" not in text.replace("$ ", "")


FUSED_SNAPSHOT = Path(__file__).parents[1] / "data" / "card_fused_snapshot.md"


def test_fused_render_matches_snapshot():
    """The fused card; regenerate with ``UPDATE_SNAPSHOTS=1`` after reviewing a change."""
    text = card.render(_fused())
    if os.environ.get("UPDATE_SNAPSHOTS"):
        FUSED_SNAPSHOT.write_text(text)
    assert text == FUSED_SNAPSHOT.read_text()


def test_plain_card_has_no_scene_text():
    assert card.scene_parts(_context()) == dict.fromkeys(card.SCENE_PARTS, "")
    assert "scene" not in card.render(_context()).lower()


def test_scene_template_has_every_part():
    assert tuple(card.parts(card.SCENES_TEMPLATE.read_text())) == card.SCENE_PARTS


def test_parts():
    assert card.parts("<!-- a -->\nx\n\n<!-- b -->\n y\n") == {"a": "x\n", "b": " y"}


def test_pinned_and_km():
    assert (
        card.pinned("me/x", "abcdef0123")
        == "[`me/x` @ `abcdef0`](https://huggingface.co/datasets/me/x/tree/abcdef0123)"
    )
    assert card.km(1000) == "1,000 km"


def test_example_language_prefers_french_else_first():
    def example(partitions):
        ctx = _context()
        text = card.render(card.Context(**{**ctx.__dict__, "partitions": partitions}))
        return text.split('load_dataset("me/data", "')[1].split('"')[0]

    assert example([("s", "deu_Latn"), ("s", "fra_Latn")]) == "fra_Latn"
    assert example([("s", "eng_Latn"), ("s", "deu_Latn")]) == "deu_Latn"


def test_sources_are_listed_one_per_line():
    ctx = _context()
    text = card.render(
        card.Context(**{**ctx.__dict__, "sources": {"a/x": "1111111aaa", "b/y": "2222222bbb"}})
    )
    assert (
        "   * [`a/x` @ `1111111`](https://huggingface.co/datasets/a/x/tree/1111111aaa)\n"
        "   * [`b/y` @ `2222222`](https://huggingface.co/datasets/b/y/tree/2222222bbb)\n"
    ) in text


@pytest.mark.parametrize(
    ("m2", "text"),
    [
        (12.3, "12 m²"),
        (9_999, "9,999 m²"),
        (10_000, "1.0 ha"),
        (523_000, "52.3 ha"),
        (520_501, "52.1 ha"),
        (1_000_000, "1.0 km²"),
        (3_477_486_032_789.2, "3,477,486.0 km²"),
    ],
)
def test_area(m2, text):
    assert card.area(m2) == text


_AXES = {1: ("towns", "farms"), 2: ("coast", "mountains"), 3: ("old", "new")}
_LANDUSE = card.Corpus(
    name="landuse",
    source="data/*.parquet",
    folder="embeddings",
    what="relevant sentences",
    unit="sentences",
    selection="distinct sentences",
    shards=2,
    min_texts=50,
    axes=_AXES,
    languages=True,
)
_SCENES = card.Corpus(
    name="scenes",
    source="scenes/*.parquet",
    folder="scene_embeddings",
    what="scene descriptions",
    unit="descriptions",
    selection="distinct scene descriptions",
    shards=1,
    min_texts=10,
    axes=_AXES,
    languages=False,
    suffix="_scenes",
)


def _themes(r2=(0.2, 0.7, 0.1), corpus=_LANDUSE):
    return card.Themes(
        corpus=corpus,
        model="org/model",
        prompt="Instruct: x\nQuery: ",
        texts=1234,
        hexagons=56,
        explained=[0.4, 0.2, 0.1],
        language_r2=list(r2),
        warning=0.5,
    )


def test_themed_card_shows_maps_legend_and_embeddings():
    ctx = card.Context(**{**_fused().__dict__, "themes": (_themes(),)})
    text = card.render(ctx)
    parsed = DatasetCard(text)
    assert [c["config_name"] for c in parsed.data.configs][-1] == "embeddings"
    maps = text.index("assets/map_wikimedia.png")
    assert (
        maps
        < text.index("assets/map_pca.png")
        < text.index("assets/map_pc3.png")
        < text.index("## Quick start")
    )
    assert (
        "* **PC1** (red, 40.0% of the variance): towns ↔ farms. "
        "The hexagon's most frequent language explains 20% of it." in text
    )
    assert "* **PC2** (green, 20.0% of the variance): coast ↔ mountains." in text
    assert text.count("Mostly language: read with care.") == 1
    assert "each of the 56 2° hexagons holding 50 sentences or more" in text
    assert "https://github.com/me/code/blob/master/docs/pca_axes.md" in text
    assert "embeddings/train-*.parquet                 relevant sentences, one vector each" in text
    assert "`scenes`, `scene_photos`, `embeddings`." in text
    assert "\n8. **Embeddings of the relevant sentences.** The 1,234 distinct sentences" in text
    assert "`Instruct: x\\nQuery: `" in text
    assert "$" not in text.replace("$ ", "")
    assert "\n\n\n" not in text.split("---", 2)[2].replace("```\n\n", "")


def test_themes_step_follows_the_last_method_step():
    text = card.render(card.Context(**{**_context().__dict__, "themes": (_themes(),)}))
    assert "Code: https://github.com/me/code\n7. **Embeddings of the relevant sentences.**" in text


def test_card_with_both_pcas():
    both = (_themes(), _themes(corpus=_SCENES))
    text = card.render(card.Context(**{**_fused().__dict__, "themes": both}))
    configs = [c["config_name"] for c in DatasetCard(text).data.configs]
    assert configs[-2:] == ["embeddings", "scene_embeddings"]
    # Each PCA sits after its corpus' density map: the sentences' in the overview, the scenes' in Scenes.
    assert text.index("assets/map_pc3.png") < text.index("## Quick start") < text.index("## Scenes")
    assert (
        text.index("assets/map_scenes.png")
        < text.index("assets/map_pca_scenes.png")
        < text.index("assets/map_pc3_scenes.png")
        < text.index("A description belongs")
    )
    assert "the 56 2° hexagons holding 10 descriptions or more" in text
    assert "docs/pca_axes_scenes.md" in text
    assert "scene_embeddings/train-*.parquet           scene descriptions, one vector each" in text
    assert "`embeddings`, `scene_embeddings`." in text
    assert "**Embeddings of the relevant sentences.**" in text
    assert "\n9. **Embeddings of the scene descriptions.** The 1,234 distinct scene descriptions" in text
    # The language check is shown for the sentences only.
    assert text.count("most frequent language explains") == 3


def test_card_without_themes_has_no_embeddings():
    names = ("section_landuse", "section_scenes", "files", "configs", "method")
    assert card.theme_parts(_fused()) == dict.fromkeys(names, "")
    text = card.render(_fused())
    assert "embeddings" not in text and "map_pca" not in text


def test_themes_template_has_every_part():
    assert tuple(card.parts(card.THEMES_TEMPLATE.read_text())) == card.THEME_PARTS
