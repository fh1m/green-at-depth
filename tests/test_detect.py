import cv2
import pytest

from depthsight.detect import find_arrow
from depthsight.scene import make_scene
from depthsight.water import submerge


@pytest.mark.parametrize(
    "transform, expected",
    [
        (lambda s: s, "right"),
        (lambda s: cv2.flip(s, 1), "left"),
        (lambda s: cv2.rotate(s, cv2.ROTATE_90_CLOCKWISE), "down"),
        (lambda s: cv2.rotate(s, cv2.ROTATE_90_COUNTERCLOCKWISE), "up"),
    ],
)
def test_direction_follows_the_head_not_the_centroid(transform, expected):
    scene, _ = make_scene()
    assert find_arrow(transform(scene))[1] == expected


def test_green_looking_gate_post_is_not_an_arrow():
    # At 2 m the orange post passes the colour threshold; the shape check must reject it.
    scene, _ = make_scene()
    found = find_arrow(submerge(scene, 2.0))
    assert found is not None and found[1] == "right"
    x, y, w, h = cv2.boundingRect(found[0])
    assert x + w < 460  # the post sits at x = 470..500


def test_the_2023_desk_arrow():
    # The real mask frame from 2023: a hollow arrow pointing left, plus reflection debris.
    img = cv2.imread(str(__import__("pathlib").Path(__file__).parent.parent / "assets/original/desk-2023-arrow.png"))
    assert find_arrow(img)[1] == "left"
