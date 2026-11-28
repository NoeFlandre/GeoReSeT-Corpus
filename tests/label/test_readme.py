from pathlib import Path

import pytest

from georeset.label import readme

FIXTURE = Path(__file__).parent / "fixtures" / "corpus-README.txt"
ROUND = "lfm25-dspark-p2fb4856"


def test_card_drops_the_landuse_column_and_points_at_the_round():
    card = readme.migrate(FIXTURE.read_text(encoding="utf-8"), ROUND)
    assert "| `landuse` | bool |" not in card
    assert "s.landuse" not in card
    assert 'select_columns(["id", "text_en", "translation"])' in card
    assert f"labels/{ROUND}/decisions-*.parquet" in card
    assert f'"config_name": "labels-{ROUND}"' in card
    assert f'"config_name": "generations-{ROUND}"' in card
    assert card.index("## Labels") < card.index("## Sentences")


def test_card_refuses_a_card_that_differs_from_the_expected_text():
    with pytest.raises(ValueError, match="expected one match"):
        readme.migrate("# A different card\n", ROUND)
