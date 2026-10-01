"""OCR 따라하기 — 전처리 함수 모음 (03장).

모든 함수는 OpenCV 이미지(BGR 또는 흑백 numpy 배열)를 받아 같은 형식으로 돌려줍니다.
"""
import cv2
import numpy as np


def to_gray(img):
    """컬러 이미지를 흑백으로 바꿉니다. 이미 흑백이면 그대로 돌려줍니다."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def upscale(img, min_height=1000, max_scale=3.0):
    """세로가 min_height보다 작으면 비율을 유지하며 키웁니다 (작은 글자 확대)."""
    h = img.shape[0]
    if h >= min_height:
        return img
    scale = min(max_scale, min_height / h)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def denoise(img, strength=10):
    """잡음 제거 (Non-local Means). 흑백·컬러 모두 처리합니다."""
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, strength, 7, 21)
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def sharpen(img, amount=1.0, sigma=1.0):
    """언샤프 마스크: 흐린 이미지를 빼서 경계를 또렷하게 만듭니다."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def binarize_otsu(img):
    """오츠(Otsu) 이진화: 전체 이미지에서 글자/배경을 가르는 문턱값 하나를 자동으로 고릅니다."""
    gray = to_gray(img)
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def binarize_adaptive(img, block=31, c=15):
    """적응형 이진화: 주변 영역마다 문턱값을 따로 계산합니다 (그림자·조명 차이에 강함)."""
    gray = to_gray(img)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)


def remove_shadow(img, kernel=31):
    """배경(그림자 포함)을 추정해 나눠서 밝기를 고르게 만듭니다."""
    gray = to_gray(img)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
    background = cv2.medianBlur(background, 21)
    norm = cv2.divide(gray, background, scale=255)
    return cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)


def estimate_skew(img, max_angle=15.0):
    """글자 픽셀이 가로로 가장 잘 정렬되는 각도를 찾습니다 (투영 프로파일 방식).

    -max_angle ~ +max_angle 범위를 0.5도 간격으로 돌려 보며, 행별 글자 픽셀 수의
    분산이 가장 큰 각도를 고릅니다. 글자줄이 수평일수록 분산이 커집니다.
    """
    gray = to_gray(img)
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = ink.shape
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 0.01, 0.5):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def rotate(img, angle, border=(255, 255, 255)):
    """이미지를 angle도(반시계 방향 양수) 돌리고, 잘리지 않도록 캔버스를 넓힙니다."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    value = border if img.ndim == 3 else 255
    return cv2.warpAffine(img, m, (nw, nh), borderValue=value, flags=cv2.INTER_CUBIC)


def deskew(img, max_angle=15.0):
    """기울기를 추정해 바로 세웁니다. (보정한 이미지, 보정 각도)를 돌려줍니다."""
    angle = estimate_skew(img, max_angle)
    return (rotate(img, angle), angle) if abs(angle) >= 0.5 else (img, 0.0)


def order_corners(pts):
    """네 꼭짓점을 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래 순서로 정렬합니다."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def find_document(img, min_area_ratio=0.3):
    """이미지에서 가장 큰 사각형(문서 영역)의 네 꼭짓점을 찾습니다. 못 찾으면 None."""
    gray = to_gray(img)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = min_area_ratio * img.shape[0] * img.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) >= area_min:
            return order_corners(approx)
    return None


def warp_document(img, corners):
    """네 꼭짓점으로 둘러싸인 문서를 정면에서 본 직사각형으로 펼칩니다 (원근 보정)."""
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    m = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, m, (width, height), flags=cv2.INTER_CUBIC)
