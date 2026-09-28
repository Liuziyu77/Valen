import json
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('boto3')
spec = importlib.util.spec_from_file_location('prepare_caption', Path(__file__).resolve().parents[2] / 'scripts/data/prepare_caption.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
read_journal = module.read_journal


def test_unicode_separators_are_caption_content(tmp_path):
    path = tmp_path / 'all.jsonl'
    records = [{'caption': 'a cat\u2028on a table\x85near a window'}, {'caption': 'a dog outside'}]
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    assert list(read_journal(path)) == records


def test_only_partial_final_record_can_be_dropped(tmp_path):
    path = tmp_path / 'all.jsonl'
    path.write_text('{"caption": "complete"}\n{"caption": "part')
    assert list(read_journal(path)) == [{'caption': 'complete'}]
    path.write_text('{"caption": "complete"}\ninvalid\n{"caption": "later"}\n')
    with pytest.raises(ValueError, match='non-final'):
        list(read_journal(path))


def test_partial_utf8_tail_preserves_complete_records(tmp_path):
    path = tmp_path / 'all.jsonl'
    path.write_bytes(b'{"caption": "complete"}\n{"caption": "\xe4\xb8')
    assert list(read_journal(path)) == [{'caption': 'complete'}]
