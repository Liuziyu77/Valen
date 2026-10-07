import json

import pytest

from valen.data.schema import candidates, read_jsonl


@pytest.mark.parametrize('bins', [11, 255])
def test_score_supports_large_ordered_counting_scales(bins):
    question = {'type': 'score', 'instructions': 'Count objects.',
                'criteria': [f'Count {i}' for i in range(bins)]}
    assert candidates(question) == [(str(i), f'Count {i}') for i in range(bins)]


@pytest.mark.parametrize('bins', [1, 256])
def test_score_rejects_unsupported_candidate_counts(bins):
    with pytest.raises(ValueError, match='2–255'):
        candidates({'type': 'score', 'instructions': 'Count objects.',
                    'criteria': [f'Count {i}' for i in range(bins)]})


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
