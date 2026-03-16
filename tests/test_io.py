import json
import pytest
import numpy as np


def test_load_json_roundtrip(tmp_path):
    data = {'key': 'value', 'num': 42, 'nested': [1, 2, 3]}
    f = tmp_path / 'test.json'
    f.write_text(json.dumps(data))
    from tricalib.utils.io import load_json
    assert load_json(str(f)) == data


def test_load_json_missing_file():
    from tricalib.utils.io import load_json
    with pytest.raises(FileNotFoundError):
        load_json('/nonexistent/path/file.json')


def test_write_json_roundtrip(tmp_path):
    from tricalib.utils.io import load_json, write_json
    data = {'a': [1, 2, 3], 'b': 'hello'}
    f = str(tmp_path / 'out.json')
    write_json(f, data)
    assert load_json(f) == data


def test_write_json_appends_extension(tmp_path):
    from tricalib.utils.io import write_json, load_json
    f = str(tmp_path / 'out')  # no .json extension
    write_json(f, {'x': 1})
    assert load_json(f + '.json') == {'x': 1}


def test_fxfycxcy_to_matrix_4_element():
    from tricalib.utils.io import fxfycxcy_to_matrix
    K = fxfycxcy_to_matrix([500.0, 400.0, 320.0, 240.0])
    assert K.shape == (3, 3)
    assert K[0, 0] == 500.0  # fx
    assert K[1, 1] == 400.0  # fy
    assert K[0, 2] == 320.0  # cx
    assert K[1, 2] == 240.0  # cy
    assert K[2, 2] == 1.0
    assert K[0, 1] == 0.0    # skew
    assert K[1, 0] == 0.0


def test_fxfycxcy_to_matrix_passthrough():
    """Non-4-element input is returned as-is."""
    from tricalib.utils.io import fxfycxcy_to_matrix
    mat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert fxfycxcy_to_matrix(mat) is mat


def test_serialize_dict_converts_arrays():
    from tricalib.utils.io import serialize_dict
    data = {'arr': np.array([1.0, 2.0, 3.0]), 'scalar': 42}
    result = serialize_dict(data)
    assert isinstance(result['arr'], list)
    assert result['arr'] == [1.0, 2.0, 3.0]
    assert result['scalar'] == 42


def test_serialize_dict_2d_array():
    from tricalib.utils.io import serialize_dict
    mat = np.eye(3)
    data = {'matrix': mat}
    result = serialize_dict(data)
    assert isinstance(result['matrix'], list)
    assert result['matrix'] == mat.tolist()
