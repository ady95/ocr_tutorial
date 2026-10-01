"""한국어 OCR 평가셋 — 원문 생성 (1회 실행, 결과는 content/에 커밋).

LLM으로 가상의 한국어 문서 원문을 만든다. 실존 인물·기업·브랜드는 쓰지 않도록 지시한다.
이미지는 render.py가 이 원문으로 만들므로, 정답(Ground Truth)은 원문 그대로다.

실행: python gen_content.py            (OPENAI_API_KEY, OPENAI_MODEL 환경변수 필요)
"""
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

OUT = Path(__file__).resolve().parent.parent / "content"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

COMMON = (
    "모든 내용은 지어낸 가상의 내용이어야 합니다. 실존 인물·기업·브랜드·기관 이름, 실제 주소, "
    "실제 전화번호를 쓰지 마세요. 전화번호가 필요하면 02-000-0000, 010-0000-0000 형식만 쓰세요. "
    "결과는 JSON 객체 하나로만 출력하세요."
)

SPECS = {
    # 인쇄 문서 30
    "doc_ko": (10, "주제: {topic}. 한국어 안내문 또는 공지문. 제목 1개와 문단 3~4개, 전체 400~600자. "
                   'JSON 형식: {{"title": "...", "paragraphs": ["...", "..."]}}',
               ["아파트 관리사무소 정기 소독 안내", "도서관 휴관 및 이용 시간 변경", "사내 보안 교육 일정",
                "주민센터 평생학습 강좌 모집", "대학 학사 일정 안내", "공원 시설 보수 공사 안내",
                "동호회 정기 모임 공지", "병원 진료 시간 변경 안내", "학교 방과후 프로그램 안내",
                "지역 축제 자원봉사자 모집"]),
    "doc_mixed": (10, "주제: {topic}. 한국어와 영어 기술 용어가 섞인 기술 문서. 영어 약어(API, GPU, CPU 등), "
                      "명령어, 버전 번호, 파일 이름이 자연스럽게 섞여야 합니다. 제목 1개와 문단 3~4개, 전체 400~600자. "
                      'JSON 형식: {{"title": "...", "paragraphs": ["...", "..."]}}',
                  ["사내 Git 저장소 사용 가이드", "Docker 컨테이너 배포 절차", "REST API 인증 방식 변경 공지",
                   "GPU 서버 사용 규칙", "Python 가상환경 설정 방법", "데이터베이스 백업 정책",
                   "CI/CD 파이프라인 개선 내용", "로그 수집 시스템 소개", "모바일 앱 업데이트 노트",
                   "클라우드 스토리지 이전 계획"]),
    "doc_numeric": (10, "주제: {topic}. 숫자가 많은 한국어 문서. 날짜(2026년 3월 15일, 2026-03-15 형식 혼용), "
                        "금액(12,500원), 백분율(3.5%), 시간(09:30), 수량, 전화번호(02-000-0000 형식)를 많이 포함. "
                        '제목 1개와 문단 3~4개, 전체 400~600자. JSON 형식: {{"title": "...", "paragraphs": ["...", "..."]}}',
                    ["관리비 부과 내역 안내", "택배 요금 인상 안내", "체육센터 강습료 및 일정", "분기 매출 실적 요약",
                     "대중교통 시간표 변경", "건강검진 일정과 비용", "학원 수강료 환불 기준", "전기요금 누진 구간 안내",
                     "행사 참가비와 일정", "적금 상품 금리 안내"]),
    # 작은 문자 10
    "small": (10, "주제: {topic}. 상품 포장이나 계약서 하단에 작게 인쇄되는 주의사항·약관 문구. 문장 6~8개, 전체 300~400자. "
                  'JSON 형식: {{"title": "...", "paragraphs": ["..."]}}',
              ["식품 보관 방법과 주의사항", "전자제품 보증 조건", "개인정보 수집 동의 안내", "의약외품 사용상 주의",
               "온라인 쇼핑 반품 규정", "주차장 이용 약관", "헬스장 회원 약관", "렌터카 대여 조건",
               "공연 관람 유의사항", "택배 파손 보상 기준"]),
    # 다단 문서 10
    "multicol": (10, "주제: {topic}. 2단 편집 소식지. 제목 1개와 소제목이 있는 기사 3개. 각 기사는 소제목과 문단 2개, "
                     '전체 700~900자. JSON 형식: {{"title": "...", "articles": [{{"heading": "...", "paragraphs": ["...", "..."]}}]}}',
                 ["사내 소식지 10월호", "마을 신문 가을호", "학교 소식지", "도서관 소식지", "동물병원 소식지",
                  "체육회 소식지", "연구소 뉴스레터", "봉사단체 소식지", "시니어센터 소식지", "청년센터 소식지"]),
    # 표 20
    "table": (20, "주제: {topic}. 한국어 표 데이터. 열 4~6개, 행 5~8개. 숫자·날짜·금액이 섞이게 하세요. "
                  '첫 열은 같은 값이 2~3행씩 연속되는 구분 값(예: 부서, 분기)으로 하세요. '
                  'JSON 형식: {{"title": "...", "header": ["...", "..."], "rows": [["...", "..."]]}}',
              ["부서별 교육 이수 현황", "분기별 매출과 이익", "지점별 재고 현황", "학년별 동아리 인원", "월별 전력 사용량",
               "제품별 불량률", "지역별 강수량", "과목별 성적 분포", "노선별 승객 수", "프로젝트별 예산 집행",
               "센터별 상담 건수", "품목별 단가 비교", "연도별 회원 수", "요일별 방문객", "팀별 근무 일정",
               "매장별 고객 만족도", "장비별 점검 일정", "등급별 요금표", "단지별 관리비", "강좌별 수강 신청 현황"]),
}


def ask(client, prompt):
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": COMMON}, {"role": "user", "content": prompt}],
    )
    return json.loads(resp.choices[0].message.content)


def main():
    client = OpenAI()
    OUT.mkdir(parents=True, exist_ok=True)
    for kind, (count, template, topics) in SPECS.items():
        path = OUT / f"{kind}.json"
        if path.exists():
            print("건너뜀 (이미 있음):", path.name)
            continue
        items = []
        for i, topic in enumerate(topics[:count], start=1):
            data = ask(client, template.format(topic=topic))
            data["id"] = f"{kind}_{i:02d}"
            data["topic"] = topic
            items.append(data)
            print(kind, i, data.get("title"))
        path.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
