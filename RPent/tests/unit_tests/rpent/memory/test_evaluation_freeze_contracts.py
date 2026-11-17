# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation reads a fixed corpus and cannot publish evidence into later tasks."""

import json

import pytest

from rpent.memory import MemoryManager
from rpent.tools.common import CommonTools


def test_evaluation_reads_snapshot_and_rejects_every_corpus_writer(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "MEMORY.md"
    source.write_text("prior experience")
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "r",
                "episode_id": "e",
                "bundle_id": "b",
                "protocol_id": "p",
                "profile_ref": "profile",
                "output_dir": str(tmp_path / "run"),
                "information_scope": "release",
                "memory_ref": "memory-v1",
            }
        )
    )
    monkeypatch.setenv("RSI_RUNTIME_CONTEXT", str(context))
    manager = MemoryManager(corpus, memory_access="inbox_write", inbox_cell_tag="test")
    source.write_text("later test leakage")
    assert manager.authorize_read(source).read_text() == "prior experience"
    delivered = CommonTools(memory=manager).read_text_file(path=str(source))
    assert delivered.data["content"] == "prior experience"
    read = json.loads((tmp_path / "run/memory_reads.jsonl").read_text())
    assert read["snapshot_ref"] == str(manager.root / "snapshot.json")
    assert read["utf8_bytes_delivered"] == len("prior experience")
    reloaded = MemoryManager(corpus)
    assert reloaded.authorize_read(source).read_text() == "prior experience"
    with pytest.raises(PermissionError, match="frozen"):
        manager.authorize_write(source)
    for operation in [
        lambda: manager.merge_memory(
            cell_tag="test", run_state_dir=tmp_path, solved=True
        ),
        manager.rebuild_index,
        lambda: manager.sync(remote_repo="no-network"),
    ]:
        with pytest.raises(PermissionError, match="frozen"):
            operation()
    assert source.read_text() == "later test leakage"


def test_evaluation_snapshot_does_not_expose_another_tasks_inbox(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    private = corpus / "_internal/inbox/other"
    private.mkdir(parents=True)
    (private / "secret.md").write_text("test feedback")
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "r",
                "episode_id": "e",
                "bundle_id": "b",
                "protocol_id": "p",
                "profile_ref": "profile",
                "output_dir": str(tmp_path / "run"),
                "information_scope": "test",
            }
        )
    )
    monkeypatch.setenv("RSI_RUNTIME_CONTEXT", str(context))
    manager = MemoryManager(corpus)
    assert not (manager.root / "_internal").exists()
    with pytest.raises(PermissionError):
        manager.authorize_read(private / "secret.md")
