from opencv_rgb_policy import check_source


def codes(source: str) -> list[str]:
    return [finding.code for finding in check_source(source)]


def test_rejects_redundant_imread_conversion() -> None:
    source = """
import cv2
image = cv2.imread(path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
model(image)
"""
    assert codes(source) == ["OPCV001"]


def test_rejects_redundant_imdecode_conversion() -> None:
    source = """
import cv2 as cv
rgb = cv.cvtColor(cv.imdecode(encoded, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
"""
    assert codes(source) == ["OPCV001"]


def test_allows_bgr_when_it_is_encoded() -> None:
    source = """
import cv2
image = cv2.imread(path, cv2.IMREAD_COLOR)
ok, encoded = cv2.imencode('.webp', image)
"""
    assert codes(source) == []


def test_allows_direct_rgb_read() -> None:
    source = """
import cv2
image = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
model(image)
"""
    assert codes(source) == []


def test_does_not_reject_rgb_conversion_when_bgr_has_another_consumer() -> None:
    source = """
import cv2
image = cv2.imread(path, cv2.IMREAD_COLOR)
cv2.imwrite('copy.png', image)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
model(rgb)
"""
    assert codes(source) == []


def test_none_guard_is_not_a_bgr_consumer() -> None:
    source = """
import cv2
image = cv2.imread(path, cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError('missing image')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
model(image)
"""
    assert codes(source) == ["OPCV001"]


def test_truthiness_guard_is_not_a_bgr_consumer() -> None:
    source = """
import cv2
image = cv2.imread(path, cv2.IMREAD_COLOR)
if not image:
    raise RuntimeError('missing image')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
model(image)
"""
    assert codes(source) == ["OPCV001"]


def test_rejects_known_rgb_passed_to_imencode() -> None:
    source = """
import cv2
image = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
ok, encoded = cv2.imencode('.webp', image)
"""
    assert codes(source) == ["OPCV002"]


def test_allows_rgb_to_bgr_before_imencode() -> None:
    source = """
import cv2
image = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
ok, encoded = cv2.imencode('.webp', bgr)
"""
    assert codes(source) == []


def test_direct_bgr_to_rgb_conversion_is_allowed_when_bgr_is_used_later() -> None:
    source = """
import cv2
image = cv2.imread(path, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ok, encoded = cv2.imencode('.webp', image)
"""
    assert codes(source) == []
