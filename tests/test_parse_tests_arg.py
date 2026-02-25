from asset_processing_service.main import parse_test_numbers


def test_parse_test_numbers_single():
    assert parse_test_numbers("1") == [1]


def test_parse_test_numbers_many():
    assert parse_test_numbers("1, 3,4") == [1, 3, 4]


def test_parse_test_numbers_empty():
    assert parse_test_numbers("") is None
