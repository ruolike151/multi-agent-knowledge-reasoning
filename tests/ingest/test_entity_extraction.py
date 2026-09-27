import json
import logging

from akr.ingest.extract import ExtractionResult, extract_chunk, write_extraction
from akr.model.client import LLMResult
from akr.model.cost import Usage
from akr.model.routing import mode_for


class FakeClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResult(
            text=json.dumps(self._payload, ensure_ascii=False),
            reasoning="",
            usage=Usage(hit_tokens=0, miss_tokens=20, output_tokens=10),
            mode=kwargs["mode"],
        )


class FakeGraphStore:
    def __init__(self):
        self.calls = []

    def upsert_entity(self, entity, *, doc_id, chunk_id):
        self.calls.append(("entity", entity.entity_id(), doc_id, chunk_id))
        return entity.entity_id()

    def upsert_relation(self, source_id, relation, target_id, *, doc_id, chunk_id, evidence=""):
        self.calls.append(("relation", relation, source_id, target_id, chunk_id))


PAYLOAD = {
    "entities": [
        {"name": "甲公司", "type": "company", "aliases": ["甲"]},
        {"name": "乙公司", "type": "company", "aliases": []},
    ],
    "relations": [
        {"source": "甲公司", "relation": "SUPPLIES", "target": "乙公司", "evidence": "年报第 12 页"}
    ],
}


def make_chunk(text="甲公司在 2024 年向乙公司供应原材料。", chunk_id="doc1#c0003"):
    from akr.ingest.chunker import Chunk

    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        source="甲公司2024年年度报告.pdf",
        ordinal=3,
        text=text,
        content_hash="0123456789abcdef",
        section_path="第四节 经营情况讨论与分析 > 主要供应商",
        page_start=12,
        page_end=12,
        char_start=100,
        char_end=140,
    )


def test_extract_chunk_uses_extract_node_mode_and_json_object():
    client = FakeClient(PAYLOAD)
    extract_chunk(client, make_chunk())
    call = client.calls[0]
    assert call["mode"] == mode_for("extract")
    assert call["json_object"] is True
    assert call["system"].startswith("你是")  # system 是稳定前缀，不掺入 chunk 文本
    assert "甲公司在 2024 年" in call["messages"][0]["content"]
    assert "甲公司在 2024 年" not in call["system"]


def test_extract_chunk_parses_entities_and_relations():
    result = extract_chunk(FakeClient(PAYLOAD), make_chunk())
    assert isinstance(result, ExtractionResult)
    assert [e.name for e in result.entities] == ["甲公司", "乙公司"]
    assert result.relations[0].relation == "SUPPLIES"


def test_write_extraction_writes_entities_before_relations():
    store = FakeGraphStore()
    result = extract_chunk(FakeClient(PAYLOAD), make_chunk())
    written = write_extraction(store, result, doc_id="doc1", chunk_id="doc1#c0003")
    kinds = [call[0] for call in store.calls]
    assert kinds == ["entity", "entity", "relation"]
    assert written == (2, 1)
    assert store.calls[2][2:] == ("company:甲公司", "company:乙公司", "doc1#c0003")


def test_write_extraction_drops_unknown_relation_type(caplog):
    payload = {
        "entities": PAYLOAD["entities"],
        "relations": PAYLOAD["relations"]
        + [{"source": "甲公司", "relation": "DROP_DATABASE", "target": "乙公司"}],
    }
    store = FakeGraphStore()
    result = extract_chunk(FakeClient(payload), make_chunk())
    with caplog.at_level(logging.WARNING):
        written = write_extraction(store, result, doc_id="doc1", chunk_id="doc1#c0003")
    assert written == (2, 1)
    assert [call[1] for call in store.calls if call[0] == "relation"] == ["SUPPLIES"]
    assert "DROP_DATABASE" in caplog.text


def test_write_extraction_replay_is_idempotent():
    store = FakeGraphStore()
    result = extract_chunk(FakeClient(PAYLOAD), make_chunk())
    write_extraction(store, result, doc_id="doc1", chunk_id="doc1#c0003")
    first = list(store.calls)
    write_extraction(store, result, doc_id="doc1", chunk_id="doc1#c0003")
    assert store.calls == first + first  # 重放同一 chunk 不产生新的实体 id / 关系三元组


def test_relation_endpoint_falls_back_to_other_type_with_warning(caplog):
    payload = {
        "entities": [{"name": "甲公司", "type": "company"}],
        "relations": [{"source": "甲公司", "relation": "SUPPLIES", "target": "丙公司"}],
    }
    store = FakeGraphStore()
    result = extract_chunk(FakeClient(payload), make_chunk())
    with caplog.at_level(logging.WARNING):
        write_extraction(store, result, doc_id="doc1", chunk_id="doc1#c0003")
    assert [call for call in store.calls if call[0] == "relation"][0][3] == "other:丙公司"
    assert "丙公司" in caplog.text
