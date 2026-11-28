"""The corpus card, updated for the sidecar layout: the labels are in ``labels/<round>/``, not in ``data/``.

Each replacement must match the card exactly once; a card that differs raises instead of being half-edited.
"""

ROUND = "{round}"
EMBEDDINGS_END = (
    '{"config_name": "embeddings_en", "data_files": [{"split": "train", "path": "embeddings_en/*.parquet"}]}]'
)
LABEL_CONFIGS = (
    '{"config_name": "labels-{round}", "data_files": [{"split": "train", '
    '"path": "labels/{round}/decisions.parquet"}]}, '
    '{"config_name": "generations-{round}", "data_files": [{"split": "train", '
    '"path": "labels/{round}/generations-*.parquet"}]}]'
)
CARD_INTRO_OLD = (
    "Every row is one sentence with a boolean `landuse` label, the language, and the location of the "
    "polygon it describes. Polygon shapes are in a separate `polygons` table."
)
CARD_INTRO_NEW = (
    "Every row is one sentence with its language and the location of the polygon it describes. Polygon "
    "shapes are in a separate `polygons` table. The labels are not in `data/`: each labelling round is a "
    "sidecar table in `labels/`, keyed by `id` (see [Labels](#labels))."
)
SQL_OLD = (
    "SELECT s.text, s.landuse, p.osm_tag, p.geometry\n"
    "FROM 'hf://datasets/NoeFlandre/GeoReSeT-Corpus/data/wikipedia/*/*.parquet' s\n"
    "JOIN 'hf://datasets/NoeFlandre/GeoReSeT-Corpus/polygons/*.parquet' p "
    "ON p.polygon_id = s.polygon_ids[1]\n"
    "WHERE s.region = 'afghanistan' AND s.landuse;"
)
SQL_NEW = (
    "SELECT s.text, l.decision, p.osm_tag, p.geometry\n"
    "FROM 'hf://datasets/NoeFlandre/GeoReSeT-Corpus/data/wikipedia/*/*.parquet' s\n"
    "JOIN 'hf://datasets/NoeFlandre/GeoReSeT-Corpus/labels/{round}/decisions.parquet' l "
    "ON l.id = s.id\n"
    "JOIN 'hf://datasets/NoeFlandre/GeoReSeT-Corpus/polygons/*.parquet' p "
    "ON p.polygon_id = s.polygon_ids[1]\n"
    "WHERE s.region = 'afghanistan' AND l.decision = 'yes';"
)
LABEL_ROW = (
    "| `landuse` | bool | The sentence describes land use, land cover or a geographic environment "
    "observable by remote sensing (LLM label) |\n"
)
LABELS_SECTION = (
    "## Labels\n\n"
    "Each labelling round is a table in `labels/<round>/`, one row per sentence, joined to `data/` on `id`. "
    "The registry `labels/rounds.json` holds each round's full configuration.\n\n"
    "| Field | Type | Description |\n"
    "|---|---|---|\n"
    "| `id` | string | Sentence id, as in `data/` |\n"
    "| `text_sha256` | string | sha256 of the sentence's text; joins to `generations` |\n"
    "| `decision` | string | `yes` (describes land use, land cover or a geographic environment "
    "observable by remote sensing), `no`, or `failed` |\n"
    "| `failure` | string | Why no usable answer was given |\n\n"
    "`generations-*.parquet` (shards) holds, once per distinct text, the model's raw output with its "
    "`finish_reason` and `truncated`. Round `{round}` is the first round: its `decision` is the label "
    "`data/` carried before this release, and its raw outputs come from the three `-landuse` repositories "
    "described in the method.\n\n"
    "## Sentences\n"
)
REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (EMBEDDINGS_END, EMBEDDINGS_END[:-1] + ", " + LABEL_CONFIGS),
    (CARD_INTRO_OLD, CARD_INTRO_NEW),
    (SQL_OLD, SQL_NEW),
    (
        'select_columns(["id", "text_en", "translation", "landuse"])',
        'select_columns(["id", "text_en", "translation"])',
    ),
    (
        "**Sentences the LLM labelled relevant** (`landuse` is true):",
        "**Sentences the LLM labelled relevant** (`decision` is `yes` in round `{round}`):",
    ),
    (LABEL_ROW, ""),
    ("## Sentences\n", LABELS_SECTION),
    (
        "with `status` and `failure_reason` instead of `landuse`:",
        "with `status` and `failure_reason` in place of a label:",
    ),
)


def migrate(text: str, round_name: str) -> str:
    """The card for ``round_name``: every replacement applied, each on exactly one match."""
    for old, new in REPLACEMENTS:
        old, new = old.replace(ROUND, round_name), new.replace(ROUND, round_name)
        found = text.count(old)
        if found != 1:
            raise ValueError(f"expected one match for {old[:60]!r}, found {found}")
        text = text.replace(old, new)
    return text
