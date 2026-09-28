import json

import pytest

from valen.data.schema import read_jsonl


@pytest.mark.parametrize('separator', ['\u0085', '\u2028', '\u2029'])
def test_jsonl_keeps_unicode_separators_inside_text(tmp_path, separator):
    record = {'group_id': 'one', 'request': {'state': f'red{separator}blue', 'questions': {
        'q': {'type': 'choice', 'instructions': f'Choose{separator}one.',
              'criteria': {'a': f'red{separator}blue', 'b': 'green'}}}}}
    path = tmp_path / 'data.jsonl'
    path.write_text(json.dumps(record, ensure_ascii=False) + '\n')
    assert read_jsonl(path) == [record]


def test_jsonl_reports_physical_line_number(tmp_path):
    path = tmp_path / 'data.jsonl'; path.write_text('\n\ninvalid\n')
    with pytest.raises(ValueError, match=r'data.jsonl:3:'):
        read_jsonl(path)
