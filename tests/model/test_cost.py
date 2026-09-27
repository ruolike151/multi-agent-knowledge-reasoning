import pytest

from akr.model.cost import USD_TO_CNY, Usage


def test_cost_matches_official_price_table_peak():
    usage = Usage(hit_tokens=1_000_000, miss_tokens=1_000_000, output_tokens=1_000_000)
    expected_usd = 0.006 + 0.30 + 1.20
    assert usage.cost_cny(peak=True) == pytest.approx(expected_usd * USD_TO_CNY)


def test_off_peak_is_exactly_half_of_peak():
    usage = Usage(hit_tokens=500_000, miss_tokens=200_000, output_tokens=100_000)
    assert usage.cost_cny(peak=False) == pytest.approx(usage.cost_cny(peak=True) / 2)


def test_from_api_reads_cache_fields():
    class Raw:
        prompt_cache_hit_tokens = 10
        prompt_cache_miss_tokens = 20
        completion_tokens = 30

    usage = Usage.from_api(Raw())
    assert (usage.hit_tokens, usage.miss_tokens, usage.output_tokens) == (10, 20, 30)
    # 与真实 API 一致：total = prompt(hit+miss) + completion
    assert usage.total_tokens == 60


def test_from_api_defaults_missing_cache_fields_to_zero():
    class Raw:
        completion_tokens = 7

    usage = Usage.from_api(Raw())
    assert usage.hit_tokens == 0
    assert usage.miss_tokens == 0
