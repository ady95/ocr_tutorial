"""01장 실습 — 편집 거리, CER, WER, 자모 단위 CER을 직접 구현합니다.

예시는 04장에서 실제로 나온 OCR 결과입니다.
실행: python cer_wer.py
"""
import unicodedata

import jiwer


def edit_distance(ref, hyp):
    """ref를 hyp로 바꾸는 데 필요한 최소 편집 횟수 (삽입·삭제·교체 각 1회).

    d[i][j] = ref의 앞 i개를 hyp의 앞 j개로 바꾸는 최소 비용
    """
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i                      # ref의 앞 i개를 모두 삭제
    for j in range(len(hyp) + 1):
        d[0][j] = j                      # hyp의 앞 j개를 모두 삽입
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            same = ref[i - 1] == hyp[j - 1]
            d[i][j] = min(
                d[i - 1][j] + 1,                      # 삭제
                d[i][j - 1] + 1,                      # 삽입
                d[i - 1][j - 1] + (0 if same else 1),  # 일치 또는 교체
            )
    return d[len(ref)][len(hyp)]


def cer(ref, hyp):
    """문자 오류율 = 문자 단위 편집 거리 ÷ 정답 문자 수."""
    return edit_distance(ref, hyp) / len(ref)


def wer(ref, hyp):
    """단어 오류율 = 단어(공백으로 나눈 어절) 단위 편집 거리 ÷ 정답 단어 수."""
    return edit_distance(ref.split(), hyp.split()) / len(ref.split())


def to_jamo(text):
    """완성형 한글을 초성·중성·종성 자모로 풉니다 (유니코드 NFD 분해). '값' → 'ᄀ', 'ᅡ', 'ᆹ'."""
    return unicodedata.normalize("NFD", text)


def jamo_cer(ref, hyp):
    """자모 단위 CER: 한 글자 안의 받침 하나만 틀려도 1글자 전체를 틀린 것으로 보지 않습니다."""
    return cer(to_jamo(ref), to_jamo(hyp))


EXAMPLES = [
    ("영수증 품목 (PaddleOCR)", "계란 10구 x1", "계란10구x1"),
    ("영수증 금액 (Tesseract)", "합계 54,600원", "합계 54,6002"),
    ("안내문 (Tesseract)", "가온동 주민센터에서 주민의 배움과 취미 활동을 돕기",
     "ALE 주민센터에서 주민의 배움과 취미 PES 돕기"),
    ("받침 하나 오류", "시설 점검과 내부 자료 정리", "시설 전검과 내부 자료 정리"),
]


def main():
    print(f"{'예시':22s} {'CER':>6s} {'WER':>6s} {'자모CER':>7s} {'jiwer CER':>9s} {'jiwer WER':>9s}")
    for name, ref, hyp in EXAMPLES:
        print(f"{name:22s} {cer(ref, hyp):6.3f} {wer(ref, hyp):6.3f} {jamo_cer(ref, hyp):7.3f}"
              f" {jiwer.cer(ref, hyp):9.3f} {jiwer.wer(ref, hyp):9.3f}")
    print()
    print("자모 분해 예:", "점검 →", " ".join(to_jamo("점검")), "/ 전검 →", " ".join(to_jamo("전검")))


if __name__ == "__main__":
    main()
