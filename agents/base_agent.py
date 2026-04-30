"""
base_agent.py — 모든 에이전트의 공통 베이스 클래스
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()


def load_settings(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BaseAgent(ABC):
    """모든 에이전트가 상속받는 베이스 클래스"""

    def __init__(self, settings: dict, episode_dir: Path, mock: bool = False):
        self.settings = settings
        self.episode_dir = episode_dir
        self.llm_cfg = settings["llm"]
        self.mock = mock
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    # ------------------------------------------------------------------ #
    # LLM 호출                                                              #
    # ------------------------------------------------------------------ #
    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        if self.mock:
            return self._mock_response(system_prompt, user_prompt)
        console.print(f"  [dim]→ Claude 호출 중... ({self.llm_cfg['model']})[/dim]")
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=self.llm_cfg["model"],
            max_tokens=self.llm_cfg["max_tokens"],
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text.strip()

    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        """Mock 모드: 에이전트 타입을 추론하여 샘플 응답 반환"""
        console.print("  [dim yellow]→ [MOCK] LLM 호출 시뮬레이션[/dim yellow]")
        agent_name = type(self).__name__

        if agent_name == "ResearchAgent":
            return self._mock_research_response()
        elif agent_name == "TopicAgent":
            return self._mock_topic_response()
        elif agent_name == "ScriptAgent":
            return self._mock_script_response()
        elif agent_name == "ReviewAgent":
            return self._mock_review_response()
        elif agent_name == "AssetAgent":
            return self._mock_asset_response()
        elif agent_name == "ProductionAgent":
            return self._mock_production_response()
        return "{}"

    # ------------------------------------------------------------------ #
    # Mock 응답 샘플                                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mock_research_response() -> str:
        from datetime import datetime, timezone
        data = {
            "searched_at": datetime.now(timezone.utc).isoformat(),
            "search_queries": [
                "foreigner life in Korea vlog",
                "living in Korea as a foreigner",
                "culture shock Korea foreigner",
                "American living in Korea",
                "European living in Korea"
            ],
            "excluded_languages": ["ko"],
            "total_found": 15,
            "transcript_stats": {"success": 7, "no_transcript": 3, "skipped": 5},
            "videos": [
                {"rank": 1, "video_id": "abc001", "title": "Things That SHOCKED Me About Living in Korea", "channel": "Dave from USA", "language": "en", "published_at": "2024-11-15T10:00:00Z", "description": "American Dave shares the most shocking things about Korean daily life including delivery, internet speed, and work culture.", "url": "https://www.youtube.com/watch?v=abc001", "views": 3200000, "likes": 145000, "comments": 8900, "search_query": "culture shock Korea foreigner", "main_topics": ["culture shock", "daily life"], "ranking_score": 0.95, "ranking_reason": "Highest views, recent upload", "has_transcript": True, "transcript_language": "en"},
                {"rank": 2, "video_id": "abc002", "title": "Living Alone in Seoul as a Foreigner — My Daily Routine", "channel": "Sarah in Seoul", "language": "en", "published_at": "2024-12-01T09:00:00Z", "description": "British Sarah shows her daily routine living alone in Seoul — convenience stores, Han River picnics, and more.", "url": "https://www.youtube.com/watch?v=abc002", "views": 2800000, "likes": 132000, "comments": 7200, "search_query": "living in Korea as a foreigner", "main_topics": ["solo living", "daily routine"], "ranking_score": 0.91, "ranking_reason": "Recent upload, high engagement", "has_transcript": True, "transcript_language": "en"},
                {"rank": 3, "video_id": "abc003", "title": "Korean Work Culture SHOCKED Me (Brazilian Perspective)", "channel": "Felipe Korea Life", "language": "en", "published_at": "2024-10-20T08:00:00Z", "description": "Brazilian Felipe talks about Korean work culture — overtime, hoesik (work dinners), and the pressure to work fast.", "url": "https://www.youtube.com/watch?v=abc003", "views": 2400000, "likes": 98000, "comments": 6100, "search_query": "expat Korea daily life", "main_topics": ["work culture", "hoesik"], "ranking_score": 0.87, "ranking_reason": "Work culture topic high interest", "has_transcript": True, "transcript_language": "en"},
                {"rank": 4, "video_id": "abc004", "title": "10 Things That AMAZED Me About Korean Convenience Stores", "channel": "Emma Seoul Diary", "language": "en", "published_at": "2025-01-10T11:00:00Z", "description": "Australian Emma discovers 10 amazing things about Korean convenience stores that don't exist back home.", "url": "https://www.youtube.com/watch?v=abc004", "views": 1900000, "likes": 87000, "comments": 5400, "search_query": "foreigner life in Korea vlog", "main_topics": ["convenience store", "culture shock"], "ranking_score": 0.83, "ranking_reason": "Convenience store content popular", "has_transcript": True, "transcript_language": "en"},
                {"rank": 5, "video_id": "abc005", "title": "I Went to a Korean Hospital — Here's What Happened", "channel": "Tom's Korea", "language": "en", "published_at": "2024-09-05T14:00:00Z", "description": "Canadian Tom visits a Korean hospital for the first time and is blown away by the low cost and fast service.", "url": "https://www.youtube.com/watch?v=abc005", "views": 1750000, "likes": 79000, "comments": 4800, "search_query": "American living in Korea", "main_topics": ["healthcare", "cost of living"], "ranking_score": 0.79, "ranking_reason": "Healthcare topic generates strong reactions", "has_transcript": True, "transcript_language": "en"},
                {"rank": 6, "video_id": "abc006", "title": "Finding an Apartment in Korea as a Foreigner (Jeonse explained)", "channel": "Lena Korea Vlog", "language": "en", "published_at": "2024-08-22T10:00:00Z", "description": "German Lena explains the confusing jeonse housing system and how she finally found her apartment in Seoul.", "url": "https://www.youtube.com/watch?v=abc006", "views": 1600000, "likes": 71000, "comments": 4200, "search_query": "European living in Korea", "main_topics": ["housing", "jeonse"], "ranking_score": 0.76, "ranking_reason": "Housing info high value", "has_transcript": True, "transcript_language": "en"},
                {"rank": 7, "video_id": "abc007", "title": "Why It's HARD to Make Korean Friends (Honest Truth)", "channel": "Mike from LA", "language": "en", "published_at": "2025-02-14T09:00:00Z", "description": "LA's Mike shares his honest experience trying to make Korean friends and the cultural barriers involved.", "url": "https://www.youtube.com/watch?v=abc007", "views": 1500000, "likes": 68000, "comments": 9800, "search_query": "foreigner life in Korea vlog", "main_topics": ["friendship", "social culture"], "ranking_score": 0.74, "ranking_reason": "Very high comment count — emotional topic", "has_transcript": True, "transcript_language": "en"},
                {"rank": 8, "video_id": "abc008", "title": "Korean Food Delivery Changed My Life (30min delivery??)", "channel": "Julia Seoul Life", "language": "en", "published_at": "2025-01-28T12:00:00Z", "description": "French Julia tries Korean food delivery apps for the first time and can't believe the 30-minute delivery time.", "url": "https://www.youtube.com/watch?v=abc008", "views": 1350000, "likes": 62000, "comments": 3800, "search_query": "culture shock Korea foreigner", "main_topics": ["food delivery", "convenience"], "ranking_score": 0.71, "ranking_reason": "Food delivery trend", "has_transcript": False, "transcript_language": None},
                {"rank": 9, "video_id": "abc009", "title": "My First Time at a Korean Jimjilbang (Sauna) 🇰🇷", "channel": "Alex Korea Adventure", "language": "en", "published_at": "2024-07-18T15:00:00Z", "description": "New Zealander Alex experiences a Korean jimjilbang (sauna) for the first time — awkward but amazing!", "url": "https://www.youtube.com/watch?v=abc009", "views": 1200000, "likes": 55000, "comments": 3400, "search_query": "foreigner life in Korea vlog", "main_topics": ["jimjilbang", "bathing culture"], "ranking_score": 0.68, "ranking_reason": "Unique Korean culture content", "has_transcript": False, "transcript_language": None},
                {"rank": 10, "video_id": "abc010", "title": "Korea's Cafe Culture — An Honest Review from a European", "channel": "Sophie in Korea", "language": "en", "published_at": "2025-03-05T10:00:00Z", "description": "Belgian Sophie reviews Korean cafe culture, study cafes, and theme cafes from a European perspective.", "url": "https://www.youtube.com/watch?v=abc010", "views": 980000, "likes": 44000, "comments": 2900, "search_query": "European living in Korea", "main_topics": ["cafe culture", "study culture"], "ranking_score": 0.64, "ranking_reason": "Cafe culture sustained interest", "has_transcript": False, "transcript_language": None}
            ],
            "trend_analysis": "Foreign-language content about Korea focuses heavily on: (1) Practical shock moments — delivery speed, internet speed, hospital costs creating strong emotional reactions. (2) Social/human connection topics — making Korean friends, workplace culture, loneliness — generating the highest comment counts. (3) Unique Korean institutions — jimjilbang, PC rooms, convenience stores — unique enough to drive curiosity clicks. Videos with personal, honest narratives ('I tried...', 'shocked me', 'honest truth') dramatically outperform informational content."
        }
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _mock_topic_response() -> str:
        data = {
            "title": "한국에서 친구 사귀기 — 외국인이 말하는 솔직한 현실",
            "subtitle": "언어 장벽부터 눈치 문화까지, 외국인이 한국인 친구를 사귀기까지의 진짜 이야기",
            "keywords": ["한국 친구", "외국인 인간관계", "한국 사회문화", "눈치 문화", "언어 장벽"],
            "target_audience": "한국 거주 외국인, 한국 유학/이민 고려자, 외국인과의 교류에 관심 있는 한국인",
            "angle": "단순한 '어렵다'는 결론이 아니라, 외국인들이 실제 transcript에서 언급한 구체적 에피소드(카카오톡 문화, 처음엔 차갑지만 친해지면 극도로 따뜻한 한국인 특성)를 중심으로 공감 + 희망적 메시지 전달",
            "hook": "'한국인들은 처음엔 차가워요. 근데 한 번 친해지면... 평생 친구가 돼요.' — 이 말을 들었을 때 제가 왜 울었는지 얘기해 드릴게요.",
            "transcript_insights": [
                "7개 영상 중 5개에서 '처음엔 접근이 어렵지만 한번 친해지면 매우 따뜻하다'는 공통 언급",
                "댓글 수 1위(9,800개) 영상이 바로 친구 사귀기 주제 — 외국인과 한국인 모두 강하게 공감",
                "convenience store·delivery 주제는 이미 다수 영상에서 다뤄짐 → 틈새 주제로 '인간관계' 선택",
                "transcript에서 반복 등장 키워드: 'shy at first', 'warm inside', 'group culture', 'hierarchy'"
            ],
            "reference_videos": [
                "Why It's HARD to Make Korean Friends (Mike from LA, 1,500,000회, 댓글 9,800개)",
                "Living Alone in Seoul as a Foreigner (Sarah in Seoul, 2,800,000회)"
            ],
            "selection_reason": "Transcript 분석 결과 '친구 사귀기' 주제가 댓글 참여도 압도적 1위이며, 7개 transcript에서 공통적으로 등장하는 핵심 감정 키워드. 기존 ep002~004와 중복 없음."
        }
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _mock_script_response() -> str:
        return """## 인트로

[자막: 한국 병원, 외국인이 가면 어떤 일이?]

여러분, 혹시 다른 나라에서 아파본 적 있으세요? 저는 미국에서 살 때 감기몸살로 병원 한 번 갔다가 청구서 보고 기절할 뻔했거든요. 그런데 한국에 와서 처음 병원을 갔을 때... 진짜로 눈을 의심했습니다. 오늘은 제가 직접 겪은 한국 의료 시스템, 솔직하게 다 얘기해 드릴게요.

## 본론 1: 충격적인 비용 차이

[B-roll: 서울 시내 병원 외관, 대기실 모습]

저 처음에 한국 와서 몸이 좀 안 좋았거든요. 근처 내과에 갔는데, 대기 시간이 얼마 안 됐어요. 진찰 받고, 약 처방까지 받았는데 총 얼마 나왔는지 아세요?

[자막: 진찰료 + 약값 = 약 8,000원~15,000원]

네, 8천 원에서 1만 5천 원 사이였어요. 커피 두 잔 값이에요. 제가 미국에서 같은 상황이면 보험 없을 때 최소 150달러, 한국 돈으로 20만 원은 나왔을 거거든요. 처음엔 뭔가 잘못된 줄 알고 두 번이나 확인했어요.

[B-roll: 약국에서 약 받는 외국인 모습]

이게 가능한 이유는 바로 한국의 건강보험 제도 때문입니다. 한국에서 3개월 이상 체류하는 외국인도 건강보험에 가입할 수 있어요. 보험료를 내면 진료비의 대부분을 보험에서 커버해 주는 거죠.

## 본론 2: 빠른 속도와 높은 접근성

[B-roll: 병원 내부, 번호표 뽑는 모습]

또 놀란 게 속도예요. 저 사는 동네에만 내과, 이비인후과, 피부과가 걸어서 5분 거리에 있어요. 미국이나 유럽은 주치의 예약하려면 몇 주 기다리는 게 보통인데, 한국은 당일에 가서 바로 진료받는 게 대부분 가능하거든요.

[자막: 한국 의원급 병원 수 = 인구 대비 세계 최고 수준]

특히 동네 의원들이 정말 많아요. 한국은 의원급 병원 수가 인구 대비 세계 최고 수준이라고 하더라고요. 덕분에 웬만한 건 큰 병원 안 가고 동네에서 다 해결이 돼요.

## 본론 3: 외국인이 알아야 할 것들

[B-roll: 외국인등록증, 건강보험증 이미지]

물론 처음엔 헷갈리는 부분도 있었어요. 외국인이 건강보험 혜택을 받으려면 외국인등록증이 있어야 하고, 지역가입자 또는 직장가입자로 가입 절차를 밟아야 해요.

[자막: 건강보험 가입 조건: 체류 기간 6개월 이상 OR 직장가입자]

저처럼 처음 왔을 때 아무것도 모르면 가입이 어렵게 느껴질 수 있어요. 하지만 구청이나 국민건강보험공단에 가면 외국어 서비스도 있고, 친절하게 안내해 줘서 생각보다 어렵지 않았어요.

[B-roll: 공단 외국어 안내 자료 이미지]

그리고 한 가지 팁을 드리자면, 약국도 꼭 활용하세요. 한국 약사들은 전문 교육을 받아서, 간단한 증상은 약국에서 추천받아 해결할 수 있거든요. 이것도 외국에서는 잘 없는 문화예요.

## 본론 4: 솔직한 단점도 있어요

[자막: 단점도 솔직하게!]

물론 단점이 없는 건 아니에요. 큰 병원은 대기 시간이 길고, 의사소통이 영어로 안 되는 경우도 있어요. 서울 외국인 클리닉 같은 데 가면 영어 되는 의료진이 있긴 한데, 비용이 좀 더 나오죠.

또 건강보험 없이 외국인이 외래 진료 받을 때는 비용이 훨씬 올라가요. 그래서 처음 오시는 분들은 꼭 빨리 건강보험 가입하시길 강추해요.

## 아웃트로

[B-roll: 병원 앞에서 브이 하는 외국인]

제가 여러 나라에서 살아봤는데, 한국 의료 시스템은 진짜 수준급이에요. 비용 대비 접근성, 서비스 속도 모두 최고라고 생각해요. 물론 처음엔 시스템 파악하는 게 좀 어렵지만, 한번 익히면 정말 든든하거든요.

[자막: 구독 & 좋아요 부탁드려요!]

오늘 영상이 도움이 됐다면 좋아요랑 구독 꼭 눌러주세요! 혹시 한국 의료 시스템에 대해 궁금한 점이나 여러분의 경험이 있으면 댓글로 남겨주세요. 다음 영상에서 또 만나요!"""

    @staticmethod
    def _mock_review_response() -> str:
        return """## 인트로

[자막: 한국 병원, 외국인이 가면 어떤 일이?]

여러분, 혹시 다른 나라에서 아파본 적 있으세요? 저는 미국에서 살 때 감기몸살로 병원 한 번 갔다가 청구서 보고 진짜 기절할 뻔했거든요. 그런데 한국에 와서 처음 병원에 갔을 때... 진짜로 눈을 의심했습니다. 오늘은 외국인이 경험한 한국 의료 시스템, 솔직하게 다 얘기해 드릴게요!

## 본론 1: 충격적인 비용 차이

[B-roll: 서울 시내 병원 외관, 환자 대기실 모습]

처음 한국에 와서 몸이 좀 안 좋았어요. 근처 내과에 갔는데, 대기 시간도 짧고 진찰 받고 약 처방까지 받았는데 총 얼마 나왔는지 아세요?

[자막: 진찰료 + 약값 = 약 8,000원~15,000원]

네, 8천 원에서 1만 5천 원 사이였어요. 커피 두 잔 값이에요! 미국이었으면 같은 상황에 보험 없을 때 최소 150달러, 한국 돈으로 20만 원 이상은 나왔을 거거든요. 처음엔 계산이 잘못된 줄 알고 두 번이나 확인했어요.

[B-roll: 약국에서 약 받는 외국인 모습]

이게 가능한 이유는 한국의 건강보험 제도 덕분이에요. 한국에서 일정 기간 이상 체류하는 외국인도 건강보험에 가입할 수 있고, 보험료를 내면 진료비 대부분을 보험에서 커버해 줘요.

## 본론 2: 빠른 속도와 높은 접근성

[B-roll: 병원 내부, 번호표 뽑는 모습]

또 정말 놀란 게 속도예요. 제가 사는 동네에만 내과, 이비인후과, 피부과가 걸어서 5분 거리에 다 있어요. 미국이나 유럽은 주치의 예약하려면 몇 주 기다리는 게 보통인데, 한국은 당일에 가서 바로 진료받는 게 대부분 가능해요.

[자막: 한국 동네 의원 수 = 인구 대비 세계 최고 수준]

동네 의원이 정말 촘촘하게 있어서 웬만한 건 큰 병원 안 가고 근처에서 다 해결이 되더라고요.

## 본론 3: 외국인이 꼭 알아야 할 것들

[B-roll: 외국인등록증, 건강보험증 이미지]

처음엔 헷갈리는 부분도 있었어요. 건강보험 혜택을 받으려면 외국인등록증이 있어야 하고, 가입 절차도 밟아야 해요.

[자막: 건강보험 가입: 체류 6개월 이상 OR 직장가입자]

구청이나 국민건강보험공단에 가면 외국어 서비스도 있고 친절하게 안내해 줘서 생각보다 어렵지 않았어요. 그리고 약국도 꼭 활용하세요! 간단한 증상은 약국에서 추천받아 해결할 수 있어서 정말 편리해요.

## 본론 4: 솔직한 단점도 있어요

[자막: 단점도 솔직하게!]

물론 단점도 있어요. 큰 종합병원은 대기 시간이 길고, 영어로 소통이 안 되는 경우도 있어요. 건강보험 없이 외래 진료를 받으면 비용이 훨씬 올라가니까, 처음 오시는 분들은 건강보험 빨리 가입하시는 걸 꼭 추천해요.

## 아웃트로

[B-roll: 병원 앞에서 엄지 척 하는 외국인]

여러 나라에서 살아봤는데, 한국 의료 시스템은 비용 대비 접근성과 서비스 속도 모두 최고예요. 처음엔 시스템 파악이 좀 어렵지만, 한번 익히면 정말 든든합니다.

[자막: 구독 & 좋아요!]

오늘 영상이 도움됐다면 좋아요와 구독 꼭 눌러주세요! 한국 의료 경험이나 궁금한 점 댓글로 남겨주세요. 다음 영상에서 또 만나요!

## 변경 사항
- 인트로 에너지 강화: 느낌표 추가, 문장 흐름 다듬기
- 본론 3 마지막 문장에 약국 활용 팁 자연스럽게 연결
- 아웃트로 동작 묘사 구체화 (브이 → 엄지 척)
- 전체적으로 구어체 어색한 부분 수정"""

    @staticmethod
    def _mock_asset_response() -> str:
        assets = [
            {"scene_id": 1, "section": "인트로", "script_excerpt": "다른 나라에서 아파본 적 있으세요?", "duration_seconds": 5, "visual_type": "b-roll", "description": "진행자가 카메라 바라보며 병원 건물 앞에 서 있는 모습", "image_prompt": "A foreigner standing in front of a Korean clinic building in Seoul, sunny day, realistic photo style", "notes": "오프닝 훅, 밝고 친근한 톤"},
            {"scene_id": 2, "section": "인트로", "script_excerpt": "청구서 보고 기절할 뻔했거든요", "duration_seconds": 4, "visual_type": "인서트이미지", "description": "미국 병원 청구서 이미지 (고액 금액 강조)", "image_prompt": "American hospital bill with high cost highlighted, close-up, realistic", "notes": "미국 vs 한국 비교 효과"},
            {"scene_id": 3, "section": "본론1", "script_excerpt": "서울 시내 병원 외관, 환자 대기실", "duration_seconds": 6, "visual_type": "b-roll", "description": "깔끔한 한국 동네 내과 외관, 간판, 입구", "image_prompt": "Clean Korean neighborhood clinic exterior, medical sign in Korean, daytime, realistic", "notes": "친근한 동네 병원 분위기"},
            {"scene_id": 4, "section": "본론1", "script_excerpt": "진찰료 + 약값 = 약 8,000원~15,000원", "duration_seconds": 5, "visual_type": "자막", "description": "화면 중앙 자막: 진찰료+약값 = 8,000원~15,000원 (큰 폰트, 노란색 강조)", "image_prompt": "Korean medical receipt showing low cost around 8000-15000 KRW, close-up", "notes": "임팩트 자막, 배경 흐리게"},
            {"scene_id": 5, "section": "본론1", "script_excerpt": "약국에서 약 받는 외국인 모습", "duration_seconds": 5, "visual_type": "b-roll", "description": "외국인이 약국 창구에서 약 받으며 미소 짓는 모습", "image_prompt": "Foreigner receiving medicine at a Korean pharmacy, smiling, bright interior, realistic", "notes": "안도감 표현"},
            {"scene_id": 6, "section": "본론2", "script_excerpt": "병원 내부, 번호표 뽑는 모습", "duration_seconds": 5, "visual_type": "b-roll", "description": "번호표 발행기에서 번호표 뽑는 손, 대기 화면", "image_prompt": "Number ticket machine in Korean hospital waiting area, patient taking ticket, realistic", "notes": "효율적인 시스템 강조"},
            {"scene_id": 7, "section": "본론2", "script_excerpt": "한국 동네 의원 수 = 인구 대비 세계 최고 수준", "duration_seconds": 4, "visual_type": "애니메이션", "description": "지도 위에 의원 아이콘이 촘촘히 표시되는 인포그래픽 애니메이션", "image_prompt": "Infographic map of Seoul with many medical clinic icons clustered, clean design", "notes": "인포그래픽 스타일, 2초 애니메이션"},
            {"scene_id": 8, "section": "본론3", "script_excerpt": "외국인등록증, 건강보험증 이미지", "duration_seconds": 5, "visual_type": "인서트이미지", "description": "외국인등록증과 건강보험증 나란히 놓인 이미지", "image_prompt": "Korean foreigner registration card and health insurance card side by side, flat lay, realistic", "notes": "중요 정보 강조"},
            {"scene_id": 9, "section": "본론3", "script_excerpt": "건강보험 가입: 체류 6개월 이상 OR 직장가입자", "duration_seconds": 5, "visual_type": "자막", "description": "화면 하단 자막 박스: 가입 조건 체크리스트", "image_prompt": "", "notes": "체크리스트 형식, 녹색 체크마크"},
            {"scene_id": 10, "section": "본론4", "script_excerpt": "단점도 솔직하게!", "duration_seconds": 3, "visual_type": "자막", "description": "화면 전환과 함께 '단점도 솔직하게!' 자막 팝업", "image_prompt": "", "notes": "화면 전환 효과 사용"},
            {"scene_id": 11, "section": "아웃트로", "script_excerpt": "병원 앞에서 엄지 척 하는 외국인", "duration_seconds": 6, "visual_type": "b-roll", "description": "진행자가 병원 앞에서 엄지 척 포즈, 밝은 표정", "image_prompt": "Foreigner giving thumbs up in front of Korean hospital entrance, happy expression, daytime", "notes": "긍정적 마무리"},
            {"scene_id": 12, "section": "아웃트로", "script_excerpt": "구독 & 좋아요!", "duration_seconds": 5, "visual_type": "자막", "description": "구독/좋아요 애니메이션 버튼 그래픽", "image_prompt": "", "notes": "유튜브 구독 버튼 애니메이션 삽입"}
        ]
        return json.dumps(assets, ensure_ascii=False)

    @staticmethod
    def _mock_production_response() -> str:
        return """# 제작 가이드: 외국인이 한국 병원에서 놀란 진짜 이유

## 1. 에피소드 개요

| 항목 | 내용 |
|------|------|
| 제목 | 외국인이 한국 병원에서 놀란 진짜 이유 |
| 부제 | 비용, 속도, 서비스... 외국인이 경험한 한국 의료 시스템의 모든 것 |
| 예상 시청 타겟 | 한국 거주 외국인, 한국 의료에 관심 있는 내국인, 해외 이민/유학 고려층 |
| 목표 시청 시간 | 8분 |

---

## 2. 편집 가이드라인

- **전체 톤**: 친근하고 정보성 강한 브이로그 스타일. 너무 딱딱하지 않게.
- **컷 편집 리듬**: 인트로·아웃트로 = 느린 컷(3~5초), 본론 = 보통(2~3초), 수치 강조 구간 = 빠른 컷
- **자막 스타일**:
  - 본문 자막: 나눔고딕 Bold, 흰색, 검정 아웃라인, 화면 하단 1/5 위치
  - 강조 자막(수치, 핵심): 노란색 또는 민트색, 크기 1.5배, 중앙 팝업
  - 섹션 전환 자막: 반투명 배경 박스 + 아이콘

---

## 3. BGM 제안

| 구간 | 분위기 | 추천 장르/키워드 |
|------|--------|-----------------|
| 인트로 (0:00~0:40) | 호기심·긴장 | Upbeat Korean indie, light electronic |
| 본론 1~2 (0:40~4:00) | 정보 전달 | Background acoustic, subtle beats |
| 본론 3~4 (4:00~7:00) | 집중 | Minimal piano, lo-fi |
| 아웃트로 (7:00~8:00) | 밝고 긍정적 | Uplifting pop, 박수 효과음 |

**추천 키워드 (Epidemic Sound/Artlist 검색용)**:
`Korea vlog`, `positive documentary`, `travel informative`, `light indie`

---

## 4. 썸네일 컨셉

### 안 A (감정 강조)
- **이미지**: 외국인이 병원 영수증 보며 눈 크게 뜨고 놀란 표정
- **텍스트**: "한국 병원 8,000원?! 😱" (빨간색 대형 폰트)
- **배경**: 병원 + 태극기 일러스트

### 안 B (비교 강조)
- **이미지**: 미국 청구서(20만원) vs 한국 청구서(8천원) 나란히
- **텍스트**: "이게 말이 돼?" (흰색 폰트, 검정 배경)
- **배경**: 분할 화면 (미국 빨강 / 한국 파랑)

---

## 5. 업로드 체크리스트

**제목 (SEO 최적화)**:
`외국인이 한국 병원에서 놀란 진짜 이유 🏥 | 비용, 건강보험, 솔직 후기`

**설명란 초안**:
```
한국에서 아프면 어떻게 될까요? 외국인의 눈으로 본 한국 의료 시스템 솔직 리뷰!
미국/유럽과 비교한 비용 차이, 건강보험 가입 방법, 꿀팁까지 모두 알려드립니다.
```

**태그 목록**:
`외국인한국생활`, `한국병원`, `한국의료`, `건강보험외국인`, `한국문화충격`,
`koreaexpat`, `koreanmedicine`, `livinginkorea`, `한국비자`, `외국인생활`,
`한국정착`, `서울생활`, `korealife`, `한국의료비`, `외국인건강보험`

**업로드 최적 시간대**:
- 한국 기준 화~목요일 오전 9시 또는 저녁 7~8시
- 주말 오전 10시 (해외 시청자 고려)

---

## 6. 제작 일정 (참고)

| 단계 | 작업 |
|------|------|
| D-1 | 스크립트 최종 확인 및 촬영 준비 |
| D-Day | 본촬영 (인터뷰 + 병원 외부 B-roll) |
| D+1 | 1차 편집 (컷 편집 + 자막) |
| D+2 | 색보정 + BGM + 효과음 |
| D+3 | 썸네일 제작 + 최종 검토 |
| D+4 | 업로드 및 SEO 설정 |
"""

    # ------------------------------------------------------------------ #
    # 파일 입출력 헬퍼                                                       #
    # ------------------------------------------------------------------ #
    def read_file(self, relative_path: str) -> str:
        path = self.episode_dir / relative_path
        return path.read_text(encoding="utf-8")

    def write_file(self, relative_path: str, content: str) -> Path:
        path = self.episode_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        console.print(f"  [green]✓[/green] 저장: {path}")
        return path

    def read_json(self, relative_path: str) -> dict | list:
        return json.loads(self.read_file(relative_path))

    def write_json(self, relative_path: str, data: dict | list) -> Path:
        return self.write_file(relative_path, json.dumps(data, ensure_ascii=False, indent=2))

    def load_template(self, template_name: str) -> str:
        tpl_dir = Path(self.settings["paths"]["templates_dir"])
        return (tpl_dir / template_name).read_text(encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 추상 메서드                                                            #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def run(self) -> None:
        """에이전트 실행 진입점"""
        ...
