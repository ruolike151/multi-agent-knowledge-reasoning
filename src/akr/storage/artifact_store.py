from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

# 只允许字母、数字、下划线、短横线、点与 #（chunk_id 形如 doc1#c3f9ab）
_SAFE = re.compile(r"^[A-Za-z0-9_#][A-Za-z0-9._#-]*$")


class ArtifactNotFound(KeyError):
    """请求的 chunk 正文不存在——必须显式失败，禁止返回空串。"""


def _safe(part: str, *, what: str) -> str:
    if not part or not _SAFE.match(part) or ".." in part:
        raise ValueError(f"非法的{what}：{part!r}（禁止路径分隔符与 ..）")
    return part


@dataclass(frozen=True)
class Artifact:
    chunk_id: str
    doc_id: str
    text: str
    char_range: tuple[int, int]
    page: int


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "chunks").mkdir(parents=True, exist_ok=True)

    def _path(self, doc_id: str, chunk_id: str) -> Path:
        doc = _safe(doc_id, what="doc_id")
        chunk = _safe(chunk_id, what="chunk_id")
        return self.root / "chunks" / doc / f"{chunk}.json"

    def put(self, artifact: Artifact) -> str:
        path = self._path(artifact.doc_id, artifact.chunk_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_id": artifact.chunk_id,
            "doc_id": artifact.doc_id,
            "page": artifact.page,
            "char_range": list(artifact.char_range),
            "text": artifact.text,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path.relative_to(self.root).as_posix()

    def get(self, doc_id: str, chunk_id: str) -> Artifact:
        path = self._path(doc_id, chunk_id)
        if not path.is_file():
            raise ArtifactNotFound(chunk_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        low, high = data["char_range"]
        return Artifact(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            text=data["text"],
            char_range=(int(low), int(high)),
            page=int(data["page"]),
        )

    def exists(self, doc_id: str, chunk_id: str) -> bool:
        return self._path(doc_id, chunk_id).is_file()

    def delete(self, doc_id: str, chunk_id: str) -> None:
        self._path(doc_id, chunk_id).unlink(missing_ok=True)

    def delete_doc(self, doc_id: str) -> int:
        doc_dir = self.root / "chunks" / _safe(doc_id, what="doc_id")
        if not doc_dir.is_dir():
            return 0
        removed = len(list(doc_dir.glob("*.json")))
        shutil.rmtree(doc_dir)
        return removed
