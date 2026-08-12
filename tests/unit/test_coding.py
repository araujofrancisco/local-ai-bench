"""Focused tests for the coding plugin's v3 evaluation machinery.

Covers: static AST anti-pattern detectors in ``score.py``, the ``raises``
test kind, relative perf scaling checks, the approach penalty, aggregate
metrics, and the optional judge blend. Canonical solutions for every case
are exercised so the dataset is self-validating.
"""

from __future__ import annotations

import asyncio

import pytest

from local_ai_bench.domain.models import (
    BenchmarkCase,
    Evaluation,
    ModelResponse,
    TimingMetrics,
    TokenMetrics,
)
from local_ai_bench.plugins.base import RunContext
from local_ai_bench.plugins.builtin.coding import CodingPlugin
from local_ai_bench.plugins.score import detect_inefficient


def _resp(text: str) -> ModelResponse:
    return ModelResponse(
        raw={},
        text=text,
        timing=TimingMetrics(total_ms=1.0),
        tokens=TokenMetrics(tokens_per_second=100.0),
    )


def _case(plugin: CodingPlugin, ctx: RunContext, case_id: str) -> BenchmarkCase:
    return next(c for c in plugin.cases(ctx) if c.id == case_id)


def _eval(plugin: CodingPlugin, case: BenchmarkCase, source: str, ctx: RunContext) -> Evaluation:
    return asyncio.run(plugin.evaluate(case, _resp(source), ctx))


# ---------------------------------------------------------------------------
# Static AST approach detectors
# ---------------------------------------------------------------------------


def test_detect_nested_loops_flags_double_loop() -> None:
    quad = (
        "def two_sum(nums, target):\n"
        "    for i in range(len(nums)):\n"
        "        for j in range(i + 1, len(nums)):\n"
        "            if nums[i] + nums[j] == target:\n"
        "                return [i, j]\n"
    )
    assert detect_inefficient(quad, ["nested_loops"]) == ["nested_loops"]


def test_detect_in_param_scan_flags_membership_scan() -> None:
    scan = (
        "def two_sum(nums, target):\n"
        "    for i in range(len(nums)):\n"
        "        if target - nums[i] in nums:\n"
        "            return [i, nums.index(target - nums[i])]\n"
    )
    assert "in_param_scan_loop" in detect_inefficient(
        scan, ["in_param_scan_loop", "linear_list_op_in_loop"]
    )
    assert "linear_list_op_in_loop" in detect_inefficient(
        scan, ["in_param_scan_loop", "linear_list_op_in_loop"]
    )


def test_detect_does_not_flag_hashmap_solution() -> None:
    fast = (
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, x in enumerate(nums):\n"
        "        need = target - x\n"
        "        if need in seen:\n"
        "            return [seen[need], i]\n"
        "        seen[x] = i\n"
    )
    assert (
        detect_inefficient(
            fast,
            ["nested_loops", "in_param_scan_loop", "linear_list_op_in_loop"],
        )
        == []
    )


def test_detect_nested_loops_fires_on_legit_dp() -> None:
    # edit_distance's optimal solution uses legitimately nested loops; the
    # `nested_loops` detector is opt-in and only attached to cases whose
    # intended solution is single-pass, so firing here just exercises the
    # detector on a real DP body.
    dp = (
        "def edit_distance(a, b):\n"
        "    n, m = len(a), len(b)\n"
        "    dp = [[0] * (m + 1) for _ in range(n + 1)]\n"
        "    for i in range(1, n + 1):\n"
        "        for j in range(1, m + 1):\n"
        "            dp[i][j] = dp[i - 1][j - 1]\n"
        "    return dp[n][m]\n"
    )
    assert detect_inefficient(dp, ["nested_loops"]) == ["nested_loops"]


def test_detect_unknown_detector_raises() -> None:
    with pytest.raises(ValueError, match="unknown approach detector"):
        detect_inefficient("def f():\n    pass\n", ["bogus"])


# ---------------------------------------------------------------------------
# Canonical solutions: the dataset is self-validating
# ---------------------------------------------------------------------------

CANONICAL: dict[str, str] = {
    "reverse_string": "def reverse_string(s):\n    return s[::-1]\n",
    "is_even": "def is_even(n):\n    return n % 2 == 0\n",
    "fizzbuzz": (
        "def fizzbuzz(n):\n"
        "    out = []\n"
        "    for i in range(1, n + 1):\n"
        "        v = ''\n"
        "        if i % 3 == 0: v += 'Fizz'\n"
        "        if i % 5 == 0: v += 'Buzz'\n"
        "        out.append(v or str(i))\n"
        "    return out\n"
    ),
    "is_anagram": (
        "def is_anagram(a, b):\n"
        "    from collections import Counter\n"
        "    return Counter(a) == Counter(b)\n"
    ),
    "is_palindrome": (
        "def is_palindrome(s):\n"
        "    t = ''.join(c for c in s.lower() if c.isalnum())\n"
        "    return t == t[::-1]\n"
    ),
    "two_sum": (
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, x in enumerate(nums):\n"
        "        need = target - x\n"
        "        if need in seen:\n"
        "            return [seen[need], i]\n"
        "        seen[x] = i\n"
    ),
    "valid_parentheses": (
        "def valid_parentheses(s):\n"
        "    st = []\n"
        "    m = {')': '(', ']': '[', '}': '{'}\n"
        "    for c in s:\n"
        "        if c in m:\n"
        "            if not st or st.pop() != m[c]: return False\n"
        "        else:\n"
        "            st.append(c)\n"
        "    return not st\n"
    ),
    "merge_intervals": (
        "def merge_intervals(intervals):\n"
        "    if not intervals: return []\n"
        "    intervals = sorted(intervals)\n"
        "    out = [intervals[0]]\n"
        "    for s, e in intervals[1:]:\n"
        "        if s <= out[-1][1]:\n"
        "            out[-1][1] = max(out[-1][1], e)\n"
        "        else:\n"
        "            out.append([s, e])\n"
        "    return out\n"
    ),
    "max_subarray": (
        "def max_subarray(nums):\n"
        "    if not nums:\n"
        "        return 0\n"
        "    best = cur = nums[0]\n"
        "    for n in nums[1:]:\n"
        "        cur = max(n, cur + n)\n"
        "        best = max(best, cur)\n"
        "    return best\n"
    ),
    "longest_common_prefix": (
        "def longest_common_prefix(strs):\n"
        "    if not strs: return ''\n"
        "    p = strs[0]\n"
        "    for s in strs[1:]:\n"
        "        while not s.startswith(p):\n"
        "            p = p[:-1]\n"
        "    return p\n"
    ),
    "LRUCache": (
        "class LRUCache:\n"
        "    def __init__(self, capacity):\n"
        "        self.cap = capacity; self.d = {}; self.order = []\n"
        "    def _touch(self, k):\n"
        "        try: self.order.remove(k)\n"
        "        except ValueError: pass\n"
        "        self.order.append(k)\n"
        "    def get(self, key):\n"
        "        if key not in self.d: return -1\n"
        "        self._touch(key); return self.d[key]\n"
        "    def put(self, key, value):\n"
        "        if key not in self.d and len(self.d) >= self.cap:\n"
        "            del self.d[self.order.pop(0)]\n"
        "        self.d[key] = value; self._touch(key)\n"
    ),
    "Trie": (
        "class Trie:\n"
        "    def __init__(self): self.root = {}\n"
        "    def insert(self, word):\n"
        "        n = self.root\n"
        "        for c in word: n = n.setdefault(c, {})\n"
        "        n['#'] = True\n"
        "    def search(self, word):\n"
        "        n = self.root\n"
        "        for c in word:\n"
        "            if c not in n: return False\n"
        "            n = n[c]\n"
        "        return n.get('#') is True\n"
        "    def starts_with(self, prefix):\n"
        "        n = self.root\n"
        "        for c in prefix:\n"
        "            if c not in n: return False\n"
        "            n = n[c]\n"
        "        return True\n"
    ),
    "edit_distance": (
        "def edit_distance(a, b):\n"
        "    n, m = len(a), len(b)\n"
        "    dp = [[0] * (m + 1) for _ in range(n + 1)]\n"
        "    for i in range(n + 1): dp[i][0] = i\n"
        "    for j in range(m + 1): dp[0][j] = j\n"
        "    for i in range(1, n + 1):\n"
        "        for j in range(1, m + 1):\n"
        "            if a[i - 1] == b[j - 1]: dp[i][j] = dp[i - 1][j - 1]\n"
        "            else: dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])\n"
        "    return dp[n][m]\n"
    ),
    "n_queens": (
        "def n_queens(n):\n"
        "    if n <= 0: return 0\n"
        "    count = 0\n"
        "    cols, d1, d2 = [False]*n, [False]*(2*n-1), [False]*(2*n-1)\n"
        "    def bt(r):\n"
        "        nonlocal count\n"
        "        if r == n:\n"
        "            count += 1; return\n"
        "        for c in range(n):\n"
        "            if cols[c] or d1[r+c] or d2[r-c+n-1]: continue\n"
        "            cols[c] = d1[r+c] = d2[r-c+n-1] = True\n"
        "            bt(r + 1)\n"
        "            cols[c] = d1[r+c] = d2[r-c+n-1] = False\n"
        "    bt(0)\n"
        "    return count\n"
    ),
    "reverse_words": "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
    "majority_element": (
        "def majority_element(nums):\n"
        "    cand = None; c = 0\n"
        "    for x in nums:\n"
        "        if c == 0: cand, c = x, 1\n"
        "        elif x == cand: c += 1\n"
        "        else: c -= 1\n"
        "    return cand\n"
    ),
    "contains_duplicate": (
        "def contains_duplicate(nums):\n    return len(set(nums)) != len(nums)\n"
    ),
    "hamming_weight": "def hamming_weight(n):\n    return bin(n).count('1')\n",
    "single_number": (
        "def single_number(nums):\n    r = 0\n    for x in nums:\n        r ^= x\n    return r\n"
    ),
    "is_power_of_two": (
        "def is_power_of_two(n):\n    return n > 0 and (n & (n - 1)) == 0\n"
    ),
    "invert_binary_tree": (
        "def invert_binary_tree(root):\n"
        "    if root is None: return None\n"
        "    root.left, root.right = invert_binary_tree(root.right), invert_binary_tree(root.left)\n"
        "    return root\n"
    ),
    "max_depth": (
        "def max_depth(root):\n"
        "    if root is None: return 0\n"
        "    return 1 + max(max_depth(root.left), max_depth(root.right))\n"
    ),
    "reverse_linked_list": (
        "def reverse_linked_list(head):\n"
        "    prev = None\n"
        "    while head:\n"
        "        nxt = head.next; head.next = prev; prev = head; head = nxt\n"
        "    return prev\n"
    ),
    "has_cycle": (
        "def has_cycle(head):\n"
        "    slow = fast = head\n"
        "    while fast and fast.next:\n"
        "        slow = slow.next; fast = fast.next.next\n"
        "        if slow is fast: return True\n"
        "    return False\n"
    ),
    "num_islands": (
        "def num_islands(grid):\n"
        "    def sink(i, j):\n"
        "        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '1': return\n"
        "        grid[i][j] = '0'\n"
        "        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)): sink(i+di, j+dj)\n"
        "    n = 0\n"
        "    for i in range(len(grid)):\n"
        "        for j in range(len(grid[0])):\n"
        "            if grid[i][j] == '1': n += 1; sink(i, j)\n"
        "    return n\n"
    ),
    "length_of_lis": (
        "def length_of_lis(nums):\n"
        "    import bisect\n"
        "    tails = []\n"
        "    for x in nums:\n"
        "        i = bisect.bisect_left(tails, x)\n"
        "        if i == len(tails): tails.append(x)\n"
        "        else: tails[i] = x\n"
        "    return len(tails)\n"
    ),
    "coin_change": (
        "def coin_change(coins, amount):\n"
        "    inf = float('inf')\n"
        "    dp = [inf] * (amount + 1); dp[0] = 0\n"
        "    for a in range(1, amount + 1):\n"
        "        for c in coins:\n"
        "            if c <= a: dp[a] = min(dp[a], dp[a - c] + 1)\n"
        "    return -1 if dp[amount] == inf else dp[amount]\n"
    ),
    "product_except_self": (
        "def product_except_self(nums):\n"
        "    n = len(nums); out = [1] * n\n"
        "    p = 1\n"
        "    for i in range(n): out[i] *= p; p *= nums[i]\n"
        "    p = 1\n"
        "    for i in range(n - 1, -1, -1): out[i] *= p; p *= nums[i]\n"
        "    return out\n"
    ),
    "group_anagrams": (
        "def group_anagrams(strs):\n"
        "    from collections import defaultdict\n"
        "    d = defaultdict(list)\n"
        "    for s in strs: d[''.join(sorted(s))].append(s)\n"
        "    return list(d.values())\n"
    ),
    "longest_substring_without_repeating": (
        "def longest_substring_without_repeating(s):\n"
        "    seen = set(); l = 0; best = 0\n"
        "    for r, ch in enumerate(s):\n"
        "        while ch in seen:\n"
        "            seen.remove(s[l]); l += 1\n"
        "        seen.add(ch); best = max(best, r - l + 1)\n"
        "    return best\n"
    ),
    "MinStack": (
        "class MinStack:\n"
        "    def __init__(self): self.s = []\n"
        "    def push(self, v): self.s.append((v, v if not self.s else min(v, self.s[-1][1])))\n"
        "    def pop(self): self.s.pop()\n"
        "    def top(self): return self.s[-1][0] if self.s else None\n"
        "    def get_min(self): return self.s[-1][1] if self.s else None\n"
    ),
    "valid_bst": (
        "def valid_bst(root, lo=float('-inf'), hi=float('inf')):\n"
        "    if root is None: return True\n"
        "    if not (lo < root.val < hi): return False\n"
        "    return valid_bst(root.left, lo, root.val) and valid_bst(root.right, root.val, hi)\n"
    ),
    "parse_ints": (
        "def parse_ints(s):\n"
        "    import re\n"
        "    toks = [t for t in re.split(r'[\\s,]+', s) if t != '']\n"
        "    out = []\n"
        "    for t in toks:\n"
        "        try: out.append(int(t))\n"
        "        except ValueError: raise ValueError('invalid int: %s' % t)\n"
        "    return out\n"
    ),
}


def _all_canonical_cases() -> list[tuple[str, str]]:
    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 60})
    out: list[tuple[str, str]] = []
    for case in plugin.cases(ctx):
        fn = case.expected["function_name"]
        sol = CANONICAL.get(fn)
        if sol:
            out.append((case.id, sol))
    return out


@pytest.mark.parametrize("case_id, source", _all_canonical_cases())
def test_canonical_solution_passes(case_id: str, source: str) -> None:
    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 60})
    case = _case(plugin, ctx, case_id)
    ev = _eval(plugin, case, source, ctx)
    assert ev.passed is True, (case_id, ev.score, ev.metrics)
    assert ev.score == 1.0


# ---------------------------------------------------------------------------
# raises test kind
# ---------------------------------------------------------------------------


def test_raises_kind_passes_when_exception_raised() -> None:
    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 30})
    case = _case(plugin, ctx, "code_parse_ints_0033")
    ev = _eval(plugin, case, CANONICAL["parse_ints"], ctx)
    # all tests including the raises assertions pass
    assert ev.passed is True
    assert ev.metrics["tests_passed"] == ev.metrics["tests_total"]


def test_raises_kind_fails_when_no_exception() -> None:
    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 30})
    case = _case(plugin, ctx, "code_parse_ints_0033")
    # a parse_ints that never raises -> the raises assertions fail
    bad = "def parse_ints(s):\n    out = []\n    for t in s.replace(',', ' ').split():\n        try:\n            out.append(int(t))\n        except ValueError:\n            pass\n    return out\n"
    ev = _eval(plugin, case, bad, ctx)
    assert ev.passed is False
    assert ev.metrics["tests_passed"] < ev.metrics["tests_total"]


# ---------------------------------------------------------------------------
# Perf scaling checks
# ---------------------------------------------------------------------------


def test_perf_passes_for_linear_solution() -> None:
    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 60})
    case = _case(plugin, ctx, "code_product_except_self_0028")
    ev = _eval(plugin, case, CANONICAL["product_except_self"], ctx)
    assert ev.metrics.get("perf_checked") is True
    assert ev.metrics.get("perf_ok") is True
    assert ev.passed is True


def test_perf_fails_for_quadratic_solution() -> None:
    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 90})
    case = _case(plugin, ctx, "code_product_except_self_0028")
    brute = (
        "def product_except_self(nums):\n"
        "    out = []\n"
        "    for i in range(len(nums)):\n"
        "        p = 1\n"
        "        for j in range(len(nums)):\n"
        "            if i != j: p *= nums[j]\n"
        "        out.append(p)\n"
        "    return out\n"
    )
    ev = _eval(plugin, case, brute, ctx)
    assert ev.metrics.get("perf_checked") is True
    assert ev.metrics.get("perf_ok") is False
    assert ev.passed is False
    # the ratio should be large (quadratic ~16x for a 4x size increase)
    assert ev.metrics.get("worst_perf_ratio", 0) > 6.0


def test_enable_perf_false_skips_perf_checks() -> None:
    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 60, "enable_perf": False})
    case = _case(plugin, ctx, "code_product_except_self_0028")
    brute = (
        "def product_except_self(nums):\n"
        "    out = []\n"
        "    for i in range(len(nums)):\n"
        "        p = 1\n"
        "        for j in range(len(nums)):\n"
        "            if i != j: p *= nums[j]\n"
        "        out.append(p)\n"
        "    return out\n"
    )
    ev = _eval(plugin, case, brute, ctx)
    assert "perf_checked" not in ev.metrics
    # brute passes the small assertions so the score is full; the nested_loops
    # approach detector then deducts the penalty rather than a perf failure.
    assert ev.metrics.get("approach_flagged") == ["nested_loops"]


# ---------------------------------------------------------------------------
# Approach penalty
# ---------------------------------------------------------------------------


def test_approach_penalty_applied_to_quadratic_two_sum_without_perf() -> None:
    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 60, "enable_perf": False})
    case = _case(plugin, ctx, "code_two_sum_0006")
    quad = (
        "def two_sum(nums, target):\n"
        "    for i in range(len(nums)):\n"
        "        for j in range(i + 1, len(nums)):\n"
        "            if nums[i] + nums[j] == target:\n"
        "                return [i, j]\n"
    )
    ev = _eval(plugin, case, quad, ctx)
    assert ev.metrics.get("approach_flagged") == ["nested_loops"]
    assert ev.metrics.get("approach_penalty_applied") is True
    assert ev.score == pytest.approx(0.9)


def test_approach_penalty_not_applied_to_optimal_solution() -> None:
    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 60, "enable_perf": False})
    case = _case(plugin, ctx, "code_two_sum_0006")
    ev = _eval(plugin, case, CANONICAL["two_sum"], ctx)
    assert "approach_flagged" not in ev.metrics
    assert ev.score == 1.0


def test_approach_penalty_not_applied_when_tests_fail() -> None:
    # a quadratic solution that is also wrong should not get the penalty on top
    # of a non-perfect score (penalty only fires at score == 1.0).
    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 60, "enable_perf": False})
    case = _case(plugin, ctx, "code_two_sum_0006")
    wrong_quad = (
        "def two_sum(nums, target):\n"
        "    for i in range(len(nums)):\n"
        "        for j in range(i + 1, len(nums)):\n"
        "            if nums[i] - nums[j] == target:\n"  # wrong op
        "                return [i, j]\n"
    )
    ev = _eval(plugin, case, wrong_quad, ctx)
    assert ev.metrics.get("approach_flagged") == ["nested_loops"]
    assert "approach_penalty_applied" not in ev.metrics
    assert ev.score < 1.0


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def test_aggregate_reports_coding_metrics() -> None:
    from local_ai_bench.domain.models import CaseResult, ModelInfo

    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 60})
    model = ModelInfo(host_name="h", model_name="m")
    two_sum = _case(plugin, ctx, "code_two_sum_0006")
    pes = _case(plugin, ctx, "code_product_except_self_0028")
    results = [
        CaseResult(
            case=two_sum,
            model=model,
            response=_resp(CANONICAL["two_sum"]),
            evaluation=_eval(plugin, two_sum, CANONICAL["two_sum"], ctx),
        ),
        CaseResult(
            case=pes,
            model=model,
            response=_resp(CANONICAL["product_except_self"]),
            evaluation=_eval(plugin, pes, CANONICAL["product_except_self"], ctx),
        ),
    ]
    agg = plugin.aggregate(results)
    assert agg.score is not None
    assert agg.metrics["pass_at_1"] == 1.0
    assert agg.metrics["perf_checked_cases"] == 2
    assert agg.metrics["complexity_ok_ratio"] == 1.0
    assert agg.metrics["syntax_ok_ratio"] == 1.0
    assert agg.metrics["tests_pass_ratio"] == 1.0


# ---------------------------------------------------------------------------
# Optional judge blend
# ---------------------------------------------------------------------------


class _StubJudge:
    """A RunContext.judge double that always returns a fixed score."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.model = "judge-m"

    async def score(self, rubric: str, input_text: str, expected: str, candidate: str) -> Evaluation | None:  # noqa: A003, ARG002
        return Evaluation(score=self.value, passed=self.value >= 0.5, rationale="stub", judge_model=self.model)


def test_judge_weight_zero_skips_judge() -> None:
    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 60, "judge_weight": 0.0})
    ctx.judge = _StubJudge(0.2)
    case = _case(plugin, ctx, "code_is_even_0002")
    ev = _eval(plugin, case, CANONICAL["is_even"], ctx)
    assert "judge_score" not in ev.metrics
    assert ev.score == 1.0


def test_judge_weight_positive_blends_score() -> None:
    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 60, "judge_weight": 0.4})
    ctx.judge = _StubJudge(0.5)
    case = _case(plugin, ctx, "code_is_even_0002")
    ev = _eval(plugin, case, CANONICAL["is_even"], ctx)
    assert ev.metrics.get("judge_score") == 0.5
    # blend: 0.6 * 1.0 + 0.4 * 0.5 = 0.8
    assert ev.score == pytest.approx(0.8, abs=1e-4)
