from collections import Counter
from pathlib import Path

import pytest

from georeset.regions import CONTINENT_OF, SUFFIX

UPSTREAM = Path(__file__).parent / "data" / "regions.txt"


@pytest.mark.parametrize(
    ("region", "continent"),
    [
        ("afghanistan", "asia"),
        ("andorra", "europe"),
        ("us-puerto-rico", "central-america"),
        ("us-texas", "north-america"),
        ("russia-siberian-fed-district", "asia"),
        ("russia-volga-fed-district", "europe"),
        ("brazil-sul", "south-america"),
        ("indonesia-papua", "oceania"),
        ("canary-islands", "africa"),
        ("antarctica", "antarctica"),
    ],
)
def test_continent(region, continent):
    assert CONTINENT_OF[region] == continent


def test_exactly_the_upstream_regions_are_mapped():
    stems = UPSTREAM.read_text().split()
    assert sorted(CONTINENT_OF) == sorted(s.removesuffix(SUFFIX) for s in stems)


def test_regions_per_continent():
    assert Counter(CONTINENT_OF.values()) == {
        "europe": 114,
        "asia": 90,
        "north-america": 67,
        "africa": 57,
        "oceania": 24,
        "south-america": 18,
        "central-america": 15,
        "antarctica": 1,
    }
