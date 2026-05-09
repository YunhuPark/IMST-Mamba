"""
Kaggle 업로드용 코드 압축 스크립트.
data/ 폴더(용량 큼)는 제외하고 코드만 압축합니다.

사용법:
    python scripts/zip_for_kaggle.py
출력:
    triage_code.zip  (프로젝트 루트에 생성)
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "triage_code.zip"

EXCLUDE_ROOT_DIRS = {"results", "__pycache__", ".pytest_cache", ".git", "node_modules"}
EXCLUDE_EXTS = {".pyc", ".pyo", ".zip"}

def should_include(path: Path) -> bool:
    parts = path.parts
    # 루트 바로 아래 data/ 폴더만 제외 (src/data/ 는 포함)
    if parts[0] == "data":
        return False
    for part in parts:
        if part in EXCLUDE_ROOT_DIRS:
            return False
    if path.suffix in EXCLUDE_EXTS:
        return False
    return True

files = [p for p in ROOT.rglob("*") if p.is_file() and should_include(p.relative_to(ROOT))]

with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        arcname = f.relative_to(ROOT)
        zf.write(f, arcname)
        print(f"  + {arcname}")

size_mb = OUTPUT.stat().st_size / 1024 / 1024
print(f"\n완료: {OUTPUT.name} ({size_mb:.1f} MB), {len(files)}개 파일")
