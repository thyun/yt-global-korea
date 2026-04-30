#!/usr/bin/env python3
"""
run.py — YouTube 동영상 제작 에이전트 오케스트레이터

파이프라인 순서:
    research → topic → script → review → asset → production

사용법:
    python run.py                        # 전 단계 자동 실행
    python run.py --step topic           # 특정 단계만 실행
    python run.py --episode ep001        # 기존 에피소드 이어서 실행
    python run.py --from-step script     # 특정 단계부터 재실행

환경 변수 (.env):
    OPENAI_API_KEY=sk-...        (필수)
    YOUTUBE_API_KEY=AIza...      (선택 — 없으면 LLM 시뮬레이션)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from agents.base_agent import load_settings
from agents.research_agent import ResearchAgent
from agents.topic_agent import TopicAgent
from agents.script_agent import ScriptAgent
from agents.review_agent import ReviewAgent
from agents.asset_agent import AssetAgent
from agents.production_agent import ProductionAgent

console = Console()

# 파이프라인 순서: research가 먼저, topic이 두 번째
STEPS = ["research", "topic", "script", "review", "asset", "production"]

AGENT_MAP = {
    "research": ResearchAgent,
    "topic": TopicAgent,
    "script": ScriptAgent,
    "review": ReviewAgent,
    "asset": AssetAgent,
    "production": ProductionAgent,
}


# ------------------------------------------------------------------ #
# 에피소드 디렉터리 관리                                                 #
# ------------------------------------------------------------------ #

def get_next_episode_id(episodes_dir: Path) -> str:
    existing = sorted([
        d.name for d in episodes_dir.iterdir()
        if d.is_dir() and re.match(r"ep\d+", d.name)
    ]) if episodes_dir.exists() else []
    if not existing:
        return "ep001"
    last_num = int(re.match(r"ep(\d+)", existing[-1]).group(1))
    return f"ep{last_num + 1:03d}"


def find_episode_dir(episodes_dir: Path, episode_id: str) -> Path | None:
    if not episodes_dir.exists():
        return None
    for d in episodes_dir.iterdir():
        if d.is_dir() and d.name.startswith(episode_id):
            return d
    return None


def create_episode_dir(episodes_dir: Path, episode_id: str, topic_data: dict) -> Path:
    from slugify import slugify
    # english_title이 있으면 영어 번역명 사용, 없으면 한국어 title 슬러그화 (로마자 변환)
    english_title = topic_data.get("english_title", "").strip()
    if english_title:
        title_slug = slugify(english_title, allow_unicode=False, max_length=50)
    else:
        title_slug = slugify(topic_data.get("title", "untitled"), allow_unicode=False, max_length=40)
    ep_dir = episodes_dir / f"{episode_id}_{title_slug}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    return ep_dir


def move_dir_contents(src: Path, dst: Path) -> None:
    """src 디렉터리의 모든 파일을 dst로 이동"""
    for f in src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(src)
            dest = dst / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            f.rename(dest)
    # 빈 디렉터리 정리
    for d in sorted(src.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass
    try:
        src.rmdir()
    except OSError:
        pass


# ------------------------------------------------------------------ #
# 메인 실행 로직                                                         #
# ------------------------------------------------------------------ #

def run_pipeline(settings: dict, episode_dir: Path, steps_to_run: list[str], mock: bool = False) -> None:
    console.print(Panel(
        f"[bold]에피소드 디렉터리:[/bold] {episode_dir}\n"
        f"[bold]실행 단계:[/bold] {' → '.join(steps_to_run)}"
        + ("\n[yellow bold][MOCK MODE] LLM 호출 없이 샘플 데이터 사용[/yellow bold]" if mock else ""),
        title="[bold green]YouTube 동영상 제작 에이전트 시작",
        expand=False,
    ))

    for step in steps_to_run:
        AgentClass = AGENT_MAP[step]
        agent = AgentClass(settings, episode_dir, mock=mock)
        try:
            agent.run()
        except Exception as e:
            console.print(f"\n[bold red]❌ {step} 단계 오류:[/bold red] {e}")
            console.print("[yellow]중단됨. --from-step 옵션으로 이 단계부터 재실행 가능합니다.[/yellow]")
            sys.exit(1)

    console.print(Panel(
        f"[bold green]✅ 모든 단계 완료![/bold green]\n\n"
        f"산출물 위치: [cyan]{episode_dir}[/cyan]",
        expand=False,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YouTube 동영상 제작 에이전트 파이프라인 (research → topic → script → review → asset → production)"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Mock 모드: LLM API 호출 없이 샘플 데이터로 전체 파이프라인 실행 (API 키 불필요)"
    )
    parser.add_argument(
        "--step", choices=STEPS,
        help="특정 단계만 실행"
    )
    parser.add_argument(
        "--from-step", choices=STEPS, dest="from_step",
        help="특정 단계부터 끝까지 실행"
    )
    parser.add_argument(
        "--episode", metavar="ID",
        help="기존 에피소드 ID (예: ep001). 미입력 시 새 에피소드 생성"
    )
    parser.add_argument(
        "--config", default="config/settings.yaml",
        help="설정 파일 경로 (기본: config/settings.yaml)"
    )
    args = parser.parse_args()
    mock: bool = args.mock

    # 설정 로드
    try:
        settings = load_settings(args.config)
    except FileNotFoundError:
        console.print(f"[red]설정 파일을 찾을 수 없습니다: {args.config}[/red]")
        sys.exit(1)

    episodes_dir = Path(settings["paths"]["episodes_dir"])
    episodes_dir.mkdir(exist_ok=True)

    # 실행할 단계 결정
    if args.step:
        steps_to_run = [args.step]
    elif args.from_step:
        start_idx = STEPS.index(args.from_step)
        steps_to_run = STEPS[start_idx:]
    else:
        steps_to_run = list(STEPS)

    # 에피소드 디렉터리 결정
    if args.episode:
        episode_dir = find_episode_dir(episodes_dir, args.episode)
        if not episode_dir:
            console.print(f"[red]에피소드를 찾을 수 없습니다: {args.episode}[/red]")
            sys.exit(1)
        console.print(f"[cyan]기존 에피소드 사용:[/cyan] {episode_dir.name}")
        run_pipeline(settings, episode_dir, steps_to_run, mock=mock)

    else:
        # 새 에피소드: research와 topic을 포함해야 함
        if "research" not in steps_to_run and "topic" not in steps_to_run:
            console.print("[red]새 에피소드 생성 시 research 또는 topic 단계가 필요합니다.[/red]")
            console.print("[yellow]기존 에피소드를 이어서 실행하려면 --episode 옵션을 사용하세요.[/yellow]")
            sys.exit(1)

        episode_id = get_next_episode_id(episodes_dir)
        temp_dir = episodes_dir / f"{episode_id}_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # research → topic 순서로 먼저 실행하여 에피소드 디렉터리 이름 결정
        bootstrap_steps = [s for s in ["research", "topic"] if s in steps_to_run]
        remaining_steps = [s for s in steps_to_run if s not in bootstrap_steps]

        console.print(Panel(
            f"[bold]새 에피소드 ID:[/bold] {episode_id}\n"
            f"[bold]부트스트랩 단계:[/bold] {' → '.join(bootstrap_steps)}\n"
            f"[bold]이후 단계:[/bold] {' → '.join(remaining_steps) or '없음'}",
            title="[bold green]새 에피소드 시작",
            expand=False,
        ))

        # 부트스트랩 단계 실행 (temp 디렉터리에서)
        for step in bootstrap_steps:
            AgentClass = AGENT_MAP[step]
            agent = AgentClass(settings, temp_dir, mock=mock)
            try:
                result = agent.run()
            except Exception as e:
                console.print(f"\n[bold red]❌ {step} 단계 오류:[/bold red] {e}")
                sys.exit(1)

        # topic 결과로 실제 에피소드 디렉터리 이름 결정
        topic_file = temp_dir / "02_topic" / "topic.json"
        if topic_file.exists():
            import json
            topic_data = json.loads(topic_file.read_text(encoding="utf-8"))
        else:
            topic_data = {"title": "untitled"}

        episode_dir = create_episode_dir(episodes_dir, episode_id, topic_data)
        move_dir_contents(temp_dir, episode_dir)
        console.print(f"  [green]✓[/green] 에피소드 디렉터리: [cyan]{episode_dir.name}[/cyan]")

        # 나머지 단계 실행
        if remaining_steps:
            run_pipeline(settings, episode_dir, remaining_steps, mock=mock)
        else:
            console.print(Panel(
                f"[bold green]✅ 완료![/bold green]\n산출물 위치: [cyan]{episode_dir}[/cyan]",
                expand=False,
            ))


if __name__ == "__main__":
    main()
