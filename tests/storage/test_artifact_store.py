import pytest

from akr.storage.artifact_store import Artifact, ArtifactNotFound, ArtifactStore


def make_store(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")


def test_put_then_get_round_trips(tmp_path):
    store = make_store(tmp_path)
    relative = store.put(
        Artifact(chunk_id="doc1#cabc", doc_id="doc1", text="正文", char_range=(3, 5), page=2)
    )
    assert relative == "chunks/doc1/doc1#cabc.json"
    got = store.get("doc1", "doc1#cabc")
    assert got.text == "正文"
    assert got.char_range == (3, 5)
    assert got.page == 2


def test_get_missing_raises(tmp_path):
    with pytest.raises(ArtifactNotFound):
        make_store(tmp_path).get("doc1", "doc1#cmissing")


def test_exists_and_delete_are_idempotent(tmp_path):
    store = make_store(tmp_path)
    store.put(Artifact(chunk_id="c1", doc_id="doc1", text="x", char_range=(0, 1), page=1))
    assert store.exists("doc1", "c1") is True
    store.delete("doc1", "c1")
    store.delete("doc1", "c1")
    assert store.exists("doc1", "c1") is False


def test_delete_doc_removes_only_that_document(tmp_path):
    store = make_store(tmp_path)
    store.put(Artifact(chunk_id="c1", doc_id="doc1", text="x", char_range=(0, 1), page=1))
    store.put(Artifact(chunk_id="c2", doc_id="doc1", text="y", char_range=(1, 2), page=1))
    store.put(Artifact(chunk_id="c3", doc_id="doc2", text="z", char_range=(0, 1), page=1))
    assert store.delete_doc("doc1") == 2
    assert store.exists("doc2", "c3") is True


@pytest.mark.parametrize("bad", ["../etc", "a/../b", "a/b", "", ".hidden", "a\\b"])
def test_unsafe_identifiers_are_rejected(tmp_path, bad):
    store = make_store(tmp_path)
    with pytest.raises(ValueError):
        store.get(bad, "c1")
    with pytest.raises(ValueError):
        store.put(Artifact(chunk_id=bad, doc_id="doc1", text="x", char_range=(0, 1), page=1))
