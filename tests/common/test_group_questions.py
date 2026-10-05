from copy import deepcopy
import hashlib
import json

import pytest

from valen.data.group_questions import group_questions
from valen.data.schema import read_jsonl


def row(state="same", group="group", qid="q", positive=True):
    return {"group_id": group, "request": {"state": state, "questions": {
        qid: {"type": "noul", "instructions": "Is it valid?"}}},
        "targets": {qid: {"probabilities": {"true": float(positive), "false": float(not positive)}}},
        "meta": {"source": "example", "record_id": "original"}}


def test_grouping_preserves_every_qa_labels_provenance_and_context(tmp_path):
    source = tmp_path / "input.jsonl"
    records = [row(), row(positive=False), row(group="another"), row(state="different"), row(), row()]
    source.write_text(''.join(json.dumps(r) + '\n' for r in records))
    original = source.read_bytes()
    output = tmp_path / "out/shared.jsonl"
    manifest = group_questions(source, output, max_questions=3)
    grouped = read_jsonl(output)
    assert manifest["input_qa"] == manifest["output_qa"] == 6
    assert len(grouped) == manifest["output_records"] == 4
    assert source.read_bytes() == original
    assert hashlib.sha256(output.read_bytes()).hexdigest() == manifest["output_sha256"]
    actual = {}
    for record in grouped:
        assert len(record["request"]["questions"]) <= 3
        for qid, origin in record["meta"]["shared_state_questions"].items():
            index = origin["input_line"] - 1
            assert record["request"]["state"] == records[index]["request"]["state"]
            assert record["group_id"] == records[index]["group_id"]
            assert record["targets"][qid] == records[index]["targets"][origin["qid"]]
            actual[index] = qid
    assert len(actual) == 6
    with pytest.raises(ValueError, match="new output"):
        group_questions(source, output)


def test_split_native_multi_question_record_preserves_unlabeled_questions(tmp_path):
    record = row()
    for index in range(1, 6):
        record["request"]["questions"][str(index)] = deepcopy(record["request"]["questions"]["q"])
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(record) + '\n')
    output = tmp_path / "shared.jsonl"
    report = group_questions(source, output, max_questions=2)
    grouped = read_jsonl(output)
    assert [len(r["request"]["questions"]) for r in grouped] == [2, 2, 2]
    assert sum(len(r["targets"]) for r in grouped) == 1
    assert report["output_qa"] == 6


def test_media_paths_survive_output_directory_change(tmp_path):
    record = row({"messages": [{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "assets/clip.mp4"}}]}]})
    record["assets"] = [{"path": "assets/clip.mp4", "sha256": "hash"}]
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(record) + '\n')
    output = tmp_path / "out/shared.jsonl"
    group_questions(source, output)
    grouped = read_jsonl(output)[0]
    expected = str(tmp_path / "assets/clip.mp4")
    assert grouped["request"]["state"]["messages"][0]["content"][0]["video_url"]["url"] == expected
    assert grouped["assets"][0]["path"] == expected


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5])
def test_invalid_question_limit(tmp_path, maximum):
    with pytest.raises(ValueError, match="max_questions"):
        group_questions(tmp_path / "input", tmp_path / "output", maximum)
