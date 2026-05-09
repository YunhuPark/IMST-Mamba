"""
PhysioNet/CinC Challenge 2019 데이터 다운로드 스크립트.

사용법:
    python scripts/download_challenge2019.py

training_setA (~20,336 files) + training_setB (~20,000 files)를
data/raw/training_setA/ 와 data/raw/training_setB/ 에 저장합니다.
"""
import urllib.request
import urllib.error
from pathlib import Path
import time
import sys

BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training"

# Set A: p000001 ~ p020336
# Set B: p100001 ~ p120000
SETS = {
    "training_setA": (1, 20336),
    "training_setB": (100001, 120000),
}

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False  # 파일 없음 (정상)
            if attempt < retries - 1:
                time.sleep(2)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
    return False


def download_set(set_name: str, start: int, end: int):
    out_dir = RAW_DIR / set_name
    out_dir.mkdir(parents=True, exist_ok=True)

    total = end - start + 1
    downloaded = 0
    skipped = 0

    print(f"\n{'='*50}")
    print(f"{set_name}: p{start:06d} ~ p{end:06d} ({total:,}개)")
    print(f"저장 경로: {out_dir}")
    print(f"{'='*50}")

    for i, pid in enumerate(range(start, end + 1)):
        filename = f"p{pid:06d}.psv"
        dest = out_dir / filename

        # 이미 있으면 스킵
        if dest.exists():
            skipped += 1
            if (i + 1) % 1000 == 0:
                print(f"  [{i+1:6d}/{total}] {downloaded:,} 다운로드, {skipped:,} 스킵...")
            continue

        url = f"{BASE_URL}/{set_name}/{filename}"
        success = download_file(url, dest)

        if success:
            downloaded += 1
        # 404는 정상 (일부 번호 없을 수 있음)

        # 진행상황 출력
        if (i + 1) % 500 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"  [{i+1:6d}/{total}] {pct:.1f}%  다운로드: {downloaded:,}  스킵: {skipped:,}")

    print(f"\n완료: {downloaded:,}개 다운로드, {skipped:,}개 기존 파일")
    return downloaded


if __name__ == "__main__":
    print("PhysioNet Challenge 2019 다운로드 시작")
    print(f"저장 위치: {RAW_DIR}")

    total_downloaded = 0
    for set_name, (start, end) in SETS.items():
        n = download_set(set_name, start, end)
        total_downloaded += n

    print(f"\n{'='*50}")
    print(f"전체 완료: {total_downloaded:,}개 파일 다운로드")
    print(f"다음 단계: python scripts/run_pipeline.py --stage data")
