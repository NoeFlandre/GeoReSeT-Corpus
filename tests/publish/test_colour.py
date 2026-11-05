import numpy as np

from georeset.publish import colour

GREYS = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])


def test_oklab_round_trip_and_known_values():
    rgb = np.array([[1.0, 0.0, 0.0], [0.2, 0.6, 0.9], *GREYS])
    np.testing.assert_allclose(colour.from_oklab(colour.to_oklab(rgb)), rgb, atol=1e-6)
    lab = colour.to_oklab(GREYS)
    np.testing.assert_allclose(lab[:2, 0], [0.0, 1.0], atol=1e-4)  # black and white lightness
    np.testing.assert_allclose(lab[:, 1:], 0, atol=1e-4)  # greys have no colour


def test_nudge_keeps_colours_at_zero_shades_and_turns():
    rgb = np.array([[0.8, 0.3, 0.2], [0.2, 0.6, 0.9]])
    np.testing.assert_allclose(colour.nudge(rgb, np.zeros((2, 2))), rgb, atol=1e-6)
    lighter = colour.nudge(rgb, np.array([[1.0, 0.0], [-1.0, 0.0]]))
    lightness = colour.to_oklab(lighter)[:, 0] - colour.to_oklab(rgb)[:, 0]
    assert lightness[0] > 0.05 and lightness[1] < -0.05
    # Muted colours turn by HUE exactly; saturated ones can leave the screen's gamut and turn less.
    muted = np.array([[0.6, 0.45, 0.4], [0.35, 0.5, 0.6]])
    turned = colour.to_oklab(colour.nudge(muted, np.array([[0.0, 1.0], [0.0, 1.0]])))
    before = colour.to_oklab(muted)
    angle = np.degrees(np.arctan2(turned[:, 2], turned[:, 1]) - np.arctan2(before[:, 2], before[:, 1]))
    np.testing.assert_allclose(angle, colour.HUE, atol=0.5)
