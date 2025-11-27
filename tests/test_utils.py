from marcel_evaluation.utils import clean_url


def test_clean_url():
    # fmt: off
    expected = "uni-marburg.de/en/studying/degree-programs/sciences/datasciencems"
    case1 = "https://www.uni-marburg.de/en/studying/degree-programs/sciences/datasciencems"
    case2 = "https://www.uni-marburg.de/en/studying/degree-programs/sciences/datasciencems/"
    assert clean_url(case1) == expected
    assert clean_url(case2) == expected

    expected = "uni-marburg.de/en/studying/degree-programs/sciences/datasciencems?a=b&foo=bar"
    case1 = "https://www.uni-marburg.de/en/studying/degree-programs/sciences/datasciencems?foo=bar&a=b"
    case2 = "http://www.uni-marburg.de/en/studying/degree-programs/sciences/datasciencems?a=b&foo=bar"
    case3 = "www.uni-marburg.de/en/studying/degree-programs/sciences/datasciencems?a=b&foo=bar"
    case4 = "uni-marburg.de/en/studying/degree-programs/sciences/datasciencems?a=b&foo=bar"
    assert clean_url(case1) == expected
    assert clean_url(case2) == expected
    assert clean_url(case3) == expected
    assert clean_url(case4) == expected
    # fmt: on
