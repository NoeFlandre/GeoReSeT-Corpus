import pytest

from georeset.regions import CONTINENT_OF


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


def test_all_386_regions_mapped_once():
    assert len(CONTINENT_OF) == 386
