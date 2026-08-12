"""Coding benchmark — correctness, complexity and approach.

Dataset ``v3`` ships a heterogeneous problem set (strings, arrays, hashing,
bit manipulation, trees, linked lists, graphs, DP, robustness and stateful
classes) whose reference solutions separate strong from weak coders.

For each case:

- ``tests`` — the executable assertions. Each item is ``(code, should_pass)``
  or ``(code, should_pass, opts)`` where ``opts`` may add ``kind``
  (``"assert"`` default or ``"raises"``), ``exc`` (exception the code must
  raise for ``raises`` items) and ``weight`` (partial-credit weighting).
- ``perf`` (optional) — relative big-O checks: the harness times the solution
  on a small and a large probe and requires ``large/small < ratio`` so an O(n^2)
  answer is distinguished from the intended linear one deterministically.
- ``approach`` (optional) — static AST anti-pattern checks (see
  ``local_ai_bench.plugins.score.detect_inefficient``). When a solution passes
  every executable check but is flagged, a configurable ``approach_penalty`` is
  deducted so writing *optimal* code is rewarded, not just working code.

Generated code runs in an isolated subprocess (``python -I``, per-case time
budget) so a broken solution cannot hang the benchmark or read the host.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from contextlib import suppress
from typing import Any, ClassVar

from local_ai_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    CaseResult,
    Evaluation,
    Modality,
    PluginAggregate,
)
from local_ai_bench.judge import judge_evaluation
from local_ai_bench.plugins.builtin._base import BaseTextPlugin
from local_ai_bench.plugins.score import (
    detect_inefficient,
    extract_python,
    python_syntax_ok,
    symbol_defined,
)

# Cross-case harness fixtures: define auxiliary node classes so tree / linked
# list cases do not depend on the model defining them. A model redefining the
# same class simply overrides ours (the interface is identical).
_TREENODE = "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val, self.left, self.right = val, left, right\n"
_LISTNODE = "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val, self.next = val, next\n"

_CASES = [
    {
        "id": "code_reverse_string_0001",
        "prompt": (
            "Write a Python function named `reverse_string` that takes a string `s` "
            "and returns the reversed string. Return only the function code, no explanation."
        ),
        "function_name": "reverse_string",
        "tests": [
            ("assert reverse_string('hello') == 'olleh'", True),
            ("assert reverse_string('') == ''", True),
            ("assert reverse_string('abc') == 'cba'", True),
            ("assert reverse_string('a  b') == 'b  a'", True),
            ("assert reverse_string('héllo') == 'olléh'", True),
        ],
    },
    {
        "id": "code_is_even_0002",
        "prompt": (
            "Write a Python function named `is_even` that takes an integer `n` and "
            "returns True if n is even, False otherwise. Return only the function code."
        ),
        "function_name": "is_even",
        "tests": [
            ("assert is_even(0) is True", True),
            ("assert is_even(1) is False", True),
            ("assert is_even(42) is True", True),
            ("assert is_even(-7) is False", True),
            ("assert is_even(10**12) is True", True),
        ],
    },
    {
        "id": "code_fizzbuzz_0003",
        "prompt": (
            "Write a Python function named `fizzbuzz` that takes an integer `n` and "
            "returns a list of strings for the numbers 1..n: 'Fizz' if divisible by 3, "
            "'Buzz' if divisible by 5, 'FizzBuzz' if divisible by both, otherwise the "
            "number itself as a string. Return only the function code."
        ),
        "function_name": "fizzbuzz",
        "tests": [
            ("assert fizzbuzz(1) == ['1']", True),
            ("assert fizzbuzz(5) == ['1','2','Fizz','4','Buzz']", True),
            ("assert fizzbuzz(15)[-1] == 'FizzBuzz'", True),
            ("assert fizzbuzz(15)[2] == 'Fizz'", True),
            ("assert fizzbuzz(0) == []", True),
        ],
    },
    {
        "id": "code_is_anagram_0004",
        "prompt": (
            "Write a Python function named `is_anagram` that takes two strings `a` and "
            "`b` and returns True if they are anagrams of each other (same characters, "
            "same counts), case-sensitive. Return only the function code."
        ),
        "function_name": "is_anagram",
        "tests": [
            ("assert is_anagram('anagram', 'nagaram') is True", True),
            ("assert is_anagram('rat', 'car') is False", True),
            ("assert is_anagram('', '') is True", True),
            ("assert is_anagram('aab', 'baa') is True", True),
            ("assert is_anagram('A', 'a') is False", True),
        ],
    },
    {
        "id": "code_is_palindrome_0005",
        "prompt": (
            "Write a Python function named `is_palindrome` that takes a string `s` and "
            "returns True if `s` is a palindrome ignoring case and non-alphanumeric "
            "characters. Return only the function code."
        ),
        "function_name": "is_palindrome",
        "tests": [
            ("assert is_palindrome('A man, a plan, a canal: Panama') is True", True),
            ("assert is_palindrome('race a car') is False", True),
            ("assert is_palindrome('') is True", True),
            ("assert is_palindrome('ab_a') is True", True),
            ("assert is_palindrome('abc') is False", True),
        ],
    },
    {
        "id": "code_two_sum_0006",
        "prompt": (
            "Write a Python function named `two_sum` that takes a list of integers "
            "`nums` and a target integer `target`, and returns a list of the two indices "
            "whose values add up to `target`, or None if no such pair exists. The function "
            "must be O(n): do not scan the list inside a loop. Return only the function code."
        ),
        "function_name": "two_sum",
        "approach": ["nested_loops", "in_param_scan_loop", "linear_list_op_in_loop"],
        "tests": [
            ("assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]", True),
            ("assert sorted(two_sum([3, 2, 4], 6)) == [1, 2]", True),
            ("assert sorted(two_sum([3, 3], 6)) == [0, 1]", True),
            ("assert sorted(two_sum([5, -2, 3], 3)) == [0, 1]", True),
            ("assert two_sum([1], 2) is None", True),
        ],
        "perf": [
            {
                "name": "two_sum_linear",
                "small": "two_sum(list(range(20000)), 19597)",
                "large": "two_sum(list(range(80000)), 79597)",
                "ratio": 6.0,
            }
        ],
    },
    {
        "id": "code_valid_parentheses_0007",
        "prompt": (
            "Write a Python function named `valid_parentheses` that takes a string `s` "
            "containing only '(', ')', '{', '}', '[' and ']' and returns True if the "
            "brackets are correctly closed in the right order. Return only the function code."
        ),
        "function_name": "valid_parentheses",
        "tests": [
            ("assert valid_parentheses('()') is True", True),
            ("assert valid_parentheses('()[]{}') is True", True),
            ("assert valid_parentheses('(]') is False", True),
            ("assert valid_parentheses('([)]') is False", True),
            ("assert valid_parentheses('{[]}') is True", True),
            ("assert valid_parentheses('(' * 100 + ')' * 100) is True", True),
            ("assert valid_parentheses('(') is False", True),
        ],
    },
    {
        "id": "code_merge_intervals_0008",
        "prompt": (
            "Write a Python function named `merge_intervals` that takes a list of "
            "intervals `intervals`, where each interval is a list [start, end], and returns "
            "a list of merged non-overlapping intervals sorted by start. The input may be "
            "unsorted. Return only the function code."
        ),
        "function_name": "merge_intervals",
        "tests": [
            ("assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]", True),
            ("assert merge_intervals([[1,4],[4,5]]) == [[1,5]]", True),
            ("assert merge_intervals([[2,3],[1,4]]) == [[1,4]]", True),
            ("assert merge_intervals([]) == []", True),
            ("assert merge_intervals([[1,4]]) == [[1,4]]", True),
        ],
    },
    {
        "id": "code_max_subarray_0009",
        "prompt": (
            "Write a Python function named `max_subarray` that takes a list of integers "
            "`nums` and returns the maximum sum of any contiguous subarray (Kadane's "
            "algorithm). For all-negative input, return the largest single element. "
            "The function must be single-pass O(n). Return only the function code."
        ),
        "function_name": "max_subarray",
        "approach": ["nested_loops"],
        "tests": [
            ("assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6", True),
            ("assert max_subarray([1]) == 1", True),
            ("assert max_subarray([5,4,-1,7,8]) == 23", True),
            ("assert max_subarray([-2,-1]) == -1", True),
            ("assert max_subarray([-1,-2,-3]) == -1", True),
            ("assert max_subarray([]) == 0", True),
        ],
    },
    {
        "id": "code_longest_common_prefix_0010",
        "prompt": (
            "Write a Python function named `longest_common_prefix` that takes a list of "
            "strings `strs` and returns the longest common prefix string, or an empty "
            "string if there is none. Return only the function code."
        ),
        "function_name": "longest_common_prefix",
        "tests": [
            ("assert longest_common_prefix(['flower','flow','flight']) == 'fl'", True),
            ("assert longest_common_prefix(['dog','racecar','car']) == ''", True),
            ("assert longest_common_prefix(['a']) == 'a'", True),
            ("assert longest_common_prefix(['']) == ''", True),
            ("assert longest_common_prefix([]) == ''", True),
            ("assert longest_common_prefix(['ab','a']) == 'a'", True),
        ],
    },
    {
        "id": "code_lru_cache_0011",
        "prompt": (
            "Write a Python class named `LRUCache` with an __init__(self, capacity) "
            "method, plus get(self, key) returning the value or -1 if missing, and "
            "put(self, key, value). The least recently used key must be evicted when "
            "capacity is exceeded. get() and put() must be O(1) average. Return only the "
            "class code, no explanation."
        ),
        "function_name": "LRUCache",
        "tests": [
            ("cache = LRUCache(2)\ncache.put(1, 1)\ncache.put(2, 2)\nassert cache.get(1) == 1", True),
            (
                "cache = LRUCache(2)\ncache.put(1, 1)\ncache.put(2, 2)\n"
                "cache.get(1)\n"
                "cache.put(3, 3)\n"
                "assert cache.get(2) == -1",
                True,
            ),
            (
                "cache = LRUCache(2)\ncache.put(1,1)\ncache.put(2,2)\ncache.get(1)\n"
                "cache.put(3,3)\nassert cache.get(2) == -1\n"
                "cache.put(4,4)\nassert cache.get(1) == -1\n"
                "assert cache.get(3) == 3\nassert cache.get(4) == 4",
                True,
            ),
            (
                "cache = LRUCache(1)\ncache.put(1,1)\nassert cache.get(1)==1\n"
                "cache.put(2,2)\nassert cache.get(1)==-1\nassert cache.get(2)==2",
                True,
            ),
            ("cache = LRUCache(2)\nassert cache.get(9) == -1", True),
        ],
    },
    {
        "id": "code_trie_0012",
        "prompt": (
            "Write a Python class named `Trie` with __init__(self), insert(self, word), "
            "search(self, word) returning True only if word was inserted whole, and "
            "starts_with(self, prefix) returning True if any inserted word starts with "
            "prefix. Return only the class code, no explanation."
        ),
        "function_name": "Trie",
        "tests": [
            (
                "t = Trie()\nt.insert('apple')\n"
                "assert t.search('apple') is True\n"
                "assert t.search('app') is False\n"
                "assert t.starts_with('app') is True\n"
                "t.insert('app')\n"
                "assert t.search('app') is True",
                True,
            ),
            ("t = Trie()\nassert t.search('') is False", True),
            (
                "t = Trie()\nt.insert('hello')\nt.insert('hell')\n"
                "assert t.starts_with('he') is True\n"
                "assert t.starts_with('world') is False",
                True,
            ),
        ],
    },
    {
        "id": "code_edit_distance_0013",
        "prompt": (
            "Write a Python function named `edit_distance` that takes two strings `a` "
            "and `b` and returns the minimum number of single-character insertions, "
            "deletions or substitutions required to change `a` into `b` (Levenshtein). "
            "Return only the function code."
        ),
        "function_name": "edit_distance",
        "tests": [
            ("assert edit_distance('horse', 'ros') == 3", True),
            ("assert edit_distance('intention', 'execution') == 5", True),
            ("assert edit_distance('', '') == 0", True),
            ("assert edit_distance('a', '') == 1", True),
            ("assert edit_distance('', 'a') == 1", True),
            ("assert edit_distance('abc', 'abc') == 0", True),
        ],
    },
    {
        "id": "code_n_queens_0014",
        "prompt": (
            "Write a Python function named `n_queens` that takes an integer `n` and "
            "returns the number of distinct ways to place n queens on an n x n board so "
            "that no two queens attack each other. Return only the function code."
        ),
        "function_name": "n_queens",
        "tests": [
            ("assert n_queens(1) == 1", True),
            ("assert n_queens(4) == 2", True),
            ("assert n_queens(5) == 10", True),
            ("assert n_queens(6) == 4", True),
            ("assert n_queens(8) == 92", True),
        ],
    },
    {
        "id": "code_reverse_words_0015",
        "prompt": (
            "Write a Python function named `reverse_words` that takes a string `s` "
            "containing words separated by spaces and returns a string with the words in "
            "reverse order, separated by a single space, with leading/trailing whitespace "
            "removed. Return only the function code."
        ),
        "function_name": "reverse_words",
        "tests": [
            ("assert reverse_words('the sky is blue') == 'blue is sky the'", True),
            ("assert reverse_words('  hello world  ') == 'world hello'", True),
            ("assert reverse_words('a') == 'a'", True),
            ("assert reverse_words('') == ''", True),
            ("assert reverse_words(' hello ') == 'hello'", True),
            ("assert reverse_words('a b c d') == 'd c b a'", True),
        ],
    },
    {
        "id": "code_majority_element_0016",
        "prompt": (
            "Write a Python function named `majority_element` that takes a list of "
            "integers `nums` (a majority element — appearing more than n/2 times — is "
            "guaranteed to exist) and returns that element. The function should be O(n) "
            "with O(1) extra space. Return only the function code."
        ),
        "function_name": "majority_element",
        "approach": ["nested_loops", "in_param_scan_loop", "linear_list_op_in_loop"],
        "tests": [
            ("assert majority_element([3, 2, 3]) == 3", True),
            ("assert majority_element([2,2,1,1,1,2,2]) == 2", True),
            ("assert majority_element([1]) == 1", True),
            ("assert majority_element([5,5,5,1,2,5,5]) == 5", True),
            ("assert majority_element([-1, -1, 2, -1, 2]) == -1", True),
        ],
    },
    {
        "id": "code_contains_duplicate_0017",
        "prompt": (
            "Write a Python function named `contains_duplicate` that takes a list of "
            "integers `nums` and returns True if any value appears at least twice, "
            "False otherwise. It must be O(n). Return only the function code."
        ),
        "function_name": "contains_duplicate",
        "approach": ["nested_loops", "in_param_scan_loop", "linear_list_op_in_loop"],
        "tests": [
            ("assert contains_duplicate([1,2,3,1]) is True", True),
            ("assert contains_duplicate([1,2,3,4]) is False", True),
            ("assert contains_duplicate([1,1,1,3,3,4,3,2,4,2]) is True", True),
            ("assert contains_duplicate([]) is False", True),
            ("assert contains_duplicate([7]) is False", True),
            ("assert contains_duplicate([10**9, -(10**9)]) is False", True),
        ],
    },
    {
        "id": "code_hamming_weight_0018",
        "prompt": (
            "Write a Python function named `hamming_weight` that takes a non-negative "
            "integer `n` and returns the number of 1-bits in its binary representation "
            "(popcount). Return only the function code."
        ),
        "function_name": "hamming_weight",
        "tests": [
            ("assert hamming_weight(0) == 0", True),
            ("assert hamming_weight(1) == 1", True),
            ("assert hamming_weight(7) == 3", True),
            ("assert hamming_weight(13) == 3", True),
            ("assert hamming_weight(8) == 1", True),
            ("assert hamming_weight(2**31 - 1) == 31", True),
            ("assert hamming_weight(2**63) == 1", True),
        ],
    },
    {
        "id": "code_single_number_0019",
        "prompt": (
            "Write a Python function named `single_number` that takes a list of integers "
            "`nums` where every element appears exactly twice except one which appears "
            "once, and returns the element that appears once. It must run in O(n) time "
            "and O(1) space. Return only the function code."
        ),
        "function_name": "single_number",
        "tests": [
            ("assert single_number([2, 2, 1]) == 1", True),
            ("assert single_number([4,1,2,1,2]) == 4", True),
            ("assert single_number([1]) == 1", True),
            ("assert single_number([0, 0, 9, 9, 7]) == 7", True),
            ("assert single_number([-5, 3, -5, 3, 42]) == 42", True),
        ],
    },
    {
        "id": "code_is_power_of_two_0020",
        "prompt": (
            "Write a Python function named `is_power_of_two` that takes an integer `n` "
            "and returns True if `n` is a power of two. Return only the function code."
        ),
        "function_name": "is_power_of_two",
        "tests": [
            ("assert is_power_of_two(1) is True", True),
            ("assert is_power_of_two(16) is True", True),
            ("assert is_power_of_two(3) is False", True),
            ("assert is_power_of_two(0) is False", True),
            ("assert is_power_of_two(-8) is False", True),
            ("assert is_power_of_two(2**30) is True", True),
            ("assert is_power_of_two(18) is False", True),
        ],
    },
    {
        "id": "code_invert_binary_tree_0021",
        "prompt": (
            "Define a TreeNode class with val, left and right attributes (default None), "
            "then write a Python function named `invert_binary_tree` that takes the root "
            "of a binary tree and returns the root of the mirrored tree, swapping the left "
            "and right children of every node (in place is fine). Return only the code."
        ),
        "function_name": "invert_binary_tree",
        "harness": _TREENODE,
        "tests": [
            (
                "n = TreeNode(1)\nn.left = TreeNode(2)\nn.right = TreeNode(3)\n"
                "r = invert_binary_tree(n)\n"
                "assert r.val == 1 and r.right.val == 2 and r.left.val == 3",
                True,
            ),
            ("assert invert_binary_tree(None) is None", True),
            (
                "n = TreeNode(1)\nr = invert_binary_tree(n)\n"
                "assert r.val == 1 and r.left is None and r.right is None",
                True,
            ),
            (
                "n = TreeNode(4)\nn.left = TreeNode(2)\nn.left.left = TreeNode(1)\n"
                "n.left.right = TreeNode(3)\nn.right = TreeNode(7)\n"
                "n.right.left = TreeNode(6)\nn.right.right = TreeNode(9)\n"
                "r = invert_binary_tree(n)\n"
                "assert (r.left.val, r.right.val) == (7, 2)\n"
                "assert (r.left.left.val, r.left.right.val) == (9, 6)\n"
                "assert (r.right.left.val, r.right.right.val) == (3, 1)",
                True,
            ),
        ],
    },
    {
        "id": "code_max_depth_0022",
        "prompt": (
            "Define a TreeNode class with val, left and right attributes (default None), "
            "then write a Python function named `max_depth` that takes the root of a "
            "binary tree and returns its maximum depth (root-to-leaf node count); the "
            "depth of an empty tree is 0. Return only the code."
        ),
        "function_name": "max_depth",
        "harness": _TREENODE,
        "tests": [
            (
                "n = TreeNode(3)\nn.left = TreeNode(9)\nn.right = TreeNode(20)\n"
                "n.right.left = TreeNode(15)\nn.right.right = TreeNode(7)\n"
                "assert max_depth(n) == 3",
                True,
            ),
            ("assert max_depth(None) == 0", True),
            ("assert max_depth(TreeNode(1)) == 1", True),
            (
                "root = TreeNode(1)\ncur = root\n"
                "for v in range(2, 7):\n    cur.left = TreeNode(v)\n    cur = cur.left\n"
                "assert max_depth(root) == 6",
                True,
            ),
        ],
    },
    {
        "id": "code_reverse_linked_list_0023",
        "prompt": (
            "Define a ListNode class with val and next attributes (default None), then "
            "write a Python function named `reverse_linked_list` that takes the head of a "
            "singly-linked list and returns the head of the reversed list. Return only "
            "the code."
        ),
        "function_name": "reverse_linked_list",
        "harness": _LISTNODE,
        "tests": [
            (
                "h = ListNode(1)\nh.next = ListNode(2)\nh.next.next = ListNode(3)\n"
                "r = reverse_linked_list(h)\n"
                "assert r.val == 3 and r.next.val == 2 and r.next.next.val == 1\n"
                "assert r.next.next.next is None",
                True,
            ),
            ("assert reverse_linked_list(None) is None", True),
            (
                "h = ListNode(7)\nr = reverse_linked_list(h)\n"
                "assert r.val == 7 and r.next is None",
                True,
            ),
            (
                "h = ListNode(1)\nh.next = ListNode(2)\n"
                "r = reverse_linked_list(h)\n"
                "assert r.val == 2 and r.next.val == 1 and r.next.next is None",
                True,
            ),
        ],
    },
    {
        "id": "code_has_cycle_0024",
        "prompt": (
            "Define a ListNode class with val and next attributes (default None), then "
            "write a Python function named `has_cycle` that takes the head of a "
            "singly-linked list and returns True if the list contains a cycle. It must "
            "use O(1) extra space. Return only the code."
        ),
        "function_name": "has_cycle",
        "harness": _LISTNODE,
        "tests": [
            (
                "a = ListNode(1); b = ListNode(2); c = ListNode(3); d = ListNode(4)\n"
                "a.next = b; b.next = c; c.next = d; d.next = b\n"
                "assert has_cycle(a) is True",
                True,
            ),
            (
                "l = ListNode(1)\nl.next = ListNode(2)\nassert has_cycle(l) is False",
                True,
            ),
            ("assert has_cycle(None) is False", True),
            (
                "s = ListNode(1)\ns.next = s\nassert has_cycle(s) is True",
                True,
            ),
        ],
    },
    {
        "id": "code_num_islands_0025",
        "prompt": (
            "Write a Python function named `num_islands` that takes a 2D list `grid` of "
            "characters '1' (land) and '0' (water) and returns the number of islands. "
            "Land connects horizontally or vertically. Water surrounds each island. It "
            "is safe to modify the input grid. Return only the function code."
        ),
        "function_name": "num_islands",
        "tests": [
            (
                "assert num_islands([['1','1','1'],['0','1','0'],['1','1','1']]) == 1",
                True,
            ),
            ("assert num_islands([]) == 0", True),
            ("assert num_islands([['1']]) == 1", True),
            ("assert num_islands([['0']]) == 0", True),
            ("assert num_islands([['1','0'],['0','1']]) == 2", True),
            (
                "assert num_islands([['1','1','0','0'],['1','0','0','0'],"
                "['0','0','1','0'],['0','0','0','1']]) == 3",
                True,
            ),
        ],
    },
    {
        "id": "code_length_of_lis_0026",
        "prompt": (
            "Write a Python function named `length_of_lis` that takes a list of integers "
            "`nums` and returns the length of the longest strictly increasing subsequence. "
            "Return only the function code."
        ),
        "function_name": "length_of_lis",
        "tests": [
            ("assert length_of_lis([10,9,2,5,3,7,101,18]) == 4", True),
            ("assert length_of_lis([0,1,0,3,2,3]) == 4", True),
            ("assert length_of_lis([7,7,7,7,7,7]) == 1", True),
            ("assert length_of_lis([]) == 0", True),
            ("assert length_of_lis([1]) == 1", True),
            ("assert length_of_lis([1,3,6,7,9,4,10,5,6]) == 6", True),
        ],
    },
    {
        "id": "code_coin_change_0027",
        "prompt": (
            "Write a Python function named `coin_change` that takes a list of coin "
            "denominations `coins` and an integer `amount` and returns the fewest number "
            "of coins needed to make up `amount`, or -1 if it is impossible. Return only "
            "the function code."
        ),
        "function_name": "coin_change",
        "tests": [
            ("assert coin_change([1, 2, 5], 11) == 3", True),
            ("assert coin_change([2], 3) == -1", True),
            ("assert coin_change([1], 0) == 0", True),
            ("assert coin_change([2, 5, 10, 1], 27) == 4", True),
            ("assert coin_change([3, 7, 8], 15) == 2", True),
            ("assert coin_change([186, 419, 83, 408], 6249) == 20", True),
        ],
    },
    {
        "id": "code_product_except_self_0028",
        "prompt": (
            "Write a Python function named `product_except_self` that takes a list of "
            "integers `nums` and returns a list where each element is the product of all "
            "the elements of the input except itself. The solution must be O(n) and must "
            "not use division. Return only the function code."
        ),
        "function_name": "product_except_self",
        "approach": ["nested_loops", "linear_list_op_in_loop"],
        "tests": [
            ("assert product_except_self([1,2,3,4]) == [24,12,8,6]", True),
            ("assert product_except_self([-1,1,0,-3,3]) == [0,0,9,0,0]", True),
            ("assert product_except_self([0,0]) == [0,0]", True),
            ("assert product_except_self([1]) == [1]", True),
            ("assert product_except_self([2,3]) == [3,2]", True),
        ],
        "perf": [
            {
                "name": "product_except_self_linear",
                "small": "product_except_self([1]*2000)",
                "large": "product_except_self([1]*8000)",
                "ratio": 6.0,
            }
        ],
    },
    {
        "id": "code_group_anagrams_0029",
        "prompt": (
            "Write a Python function named `group_anagrams` that takes a list of strings "
            "`strs` and groups anagrams together, returning a list of lists (the words of "
            "each anagram group appear together). It must run in O(n * k log k). Return "
            "only the function code."
        ),
        "function_name": "group_anagrams",
        "approach": ["nested_loops", "in_param_scan_loop", "linear_list_op_in_loop"],
        "tests": [
            (
                "g = group_anagrams(['eat','tea','tan','ate','nat','bat'])\n"
                "groups = sorted(sorted(x) for x in g)\n"
                "assert groups == [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]",
                True,
            ),
            ("assert group_anagrams(['']) == [['']]", True),
            ("assert group_anagrams(['a']) == [['a']]", True),
            ("assert group_anagrams([]) == []", True),
            (
                "g = group_anagrams(['ab','ba','abc','cba'])\n"
                "groups = sorted(sorted(x) for x in g)\n"
                "assert groups == [['ab','ba'], ['abc','cba']]",
                True,
            ),
        ],
    },
    {
        "id": "code_longest_substring_without_repeating_0030",
        "prompt": (
            "Write a Python function named `longest_substring_without_repeating` that "
            "takes a string `s` and returns the length of the longest substring without "
            "repeating characters. It must be O(n). Return only the function code."
        ),
        "function_name": "longest_substring_without_repeating",
        "approach": ["in_param_scan_loop"],
        "tests": [
            ("assert longest_substring_without_repeating('abcabcbb') == 3", True),
            ("assert longest_substring_without_repeating('bbbbb') == 1", True),
            ("assert longest_substring_without_repeating('pwwkew') == 3", True),
            ("assert longest_substring_without_repeating('') == 0", True),
            ("assert longest_substring_without_repeating(' ') == 1", True),
            ("assert longest_substring_without_repeating('abcdef') == 6", True),
            ("assert longest_substring_without_repeating('abba') == 2", True),
        ],
    },
    {
        "id": "code_min_stack_0031",
        "prompt": (
            "Write a Python class named `MinStack` supporting __init__(self), "
            "push(self, val), pop(self), top(self) and get_min(self). top(self) and "
            "get_min(self) return the top element and the minimum element of the current "
            "stack respectively, or None when the stack is empty. pop(self) removes the "
            "top element (a no-op when empty). push/get/top/pop must all be O(1). Return "
            "only the class code, no explanation."
        ),
        "function_name": "MinStack",
        "tests": [
            (
                "s = MinStack()\ns.push(-2)\ns.push(0)\ns.push(-3)\n"
                "assert s.get_min() == -3\n"
                "s.pop()\nassert s.top() == 0\nassert s.get_min() == -2",
                True,
            ),
            ("s = MinStack()\nassert s.top() is None and s.get_min() is None", True),
            ("s = MinStack()\ns.push(5)\nassert s.top() == 5 and s.get_min() == 5", True),
            (
                "s = MinStack()\nfor v in (2, 3, 1, 1):\n    s.push(v)\n"
                "assert s.get_min() == 1\n"
                "s.pop()\nassert s.get_min() == 1\n"
                "s.pop()\nassert s.get_min() == 2",
                True,
            ),
            (
                "s = MinStack()\nfor v in range(100, 0, -1):\n    s.push(v)\n"
                "assert s.top() == 1 and s.get_min() == 1",
                True,
            ),
        ],
    },
    {
        "id": "code_valid_bst_0032",
        "prompt": (
            "Define a TreeNode class with val, left and right attributes (default None), "
            "then write a Python function named `valid_bst` that takes the root of a "
            "binary tree and returns True if it is a valid binary search tree (for every "
            "node, all keys in its left subtree are strictly smaller and all keys in its "
            "right subtree strictly greater). Return only the code."
        ),
        "function_name": "valid_bst",
        "harness": _TREENODE,
        "tests": [
            (
                "r = TreeNode(2)\nr.left = TreeNode(1)\nr.right = TreeNode(3)\n"
                "assert valid_bst(r) is True",
                True,
            ),
            (
                "r = TreeNode(5)\nr.left = TreeNode(1)\nr.right = TreeNode(4)\n"
                "r.right.left = TreeNode(3)\nr.right.right = TreeNode(6)\n"
                "assert valid_bst(r) is False",
                True,
            ),
            ("assert valid_bst(None) is True", True),
            (
                "r = TreeNode(1)\nr.right = TreeNode(1)\nassert valid_bst(r) is False",
                True,
            ),
            (
                "r = TreeNode(2147483647)\nassert valid_bst(r) is True",
                True,
            ),
        ],
    },
    {
        "id": "code_parse_ints_0033",
        "prompt": (
            "Write a Python function named `parse_ints` that takes a string `s` whose "
            "tokens are integers separated by whitespace and/or commas, and returns a "
            "list of the integers. Split on commas and any whitespace. If `s` is empty or "
            "blank, return an empty list. If any token is not a valid integer, raise a "
            "ValueError with a descriptive message. Return only the function code."
        ),
        "function_name": "parse_ints",
        "tests": [
            ("assert parse_ints('1 2 3') == [1, 2, 3]", True),
            ("assert parse_ints('1,2,3') == [1, 2, 3]", True),
            ("assert parse_ints('10, 20 30') == [10, 20, 30]", True),
            ("assert parse_ints('') == []", True),
            ("assert parse_ints('   ') == []", True),
            ("assert parse_ints('-1 +2') == [-1, 2]", True),
            ("assert parse_ints('1\\n2\\n3') == [1, 2, 3]", True),
            ("assert parse_ints('3,4,') == [3, 4]", True),
            ("parse_ints('1 2 x')", True, {"kind": "raises", "exc": "ValueError"}),
            ("parse_ints('10.5')", True, {"kind": "raises", "exc": "ValueError"}),
            ("parse_ints('12a')", True, {"kind": "raises", "exc": "ValueError"}),
        ],
    },
]


class CodingPlugin(BaseTextPlugin):
    id: ClassVar[str] = "coding"
    name: ClassVar[str] = "Coding"
    description: ClassVar[str] = (
        "Correctness of generated Python against executable unit tests, with "
        "relative complexity checks and static approach analysis."
    )
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.CODING
    version: ClassVar[str] = "0.2.0"
    dataset_version: ClassVar[str] = "v3"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    _JUDGE_RUBRIC = (
        "Assess the generated Python solution against the request. Reward correct "
        "logic that handles edge cases, clean readable code, sensible naming, safe "
        "input handling, and an efficient algorithm (no avoidable quadratic work). "
        "Penalize bugs, unhandled invalid input, or needlessly slow code."
    )

    def cases(self, ctx) -> Iterable[BenchmarkCase]:  # noqa: ANN001
        for spec in _CASES:
            yield BenchmarkCase(
                id=spec["id"],
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input={"prompt": spec["prompt"]},
                expected={
                    "function_name": spec["function_name"],
                    "tests": spec["tests"],
                    **({"perf": spec["perf"]} if spec.get("perf") else {}),
                    **({"approach": spec["approach"]} if spec.get("approach") else {}),
                    **({"harness": spec["harness"]} if spec.get("harness") else {}),
                },
            )

    def build_request(self, case, model, ctx) -> dict[str, Any]:  # noqa: ANN001
        return {
            "messages": [{"role": "user", "content": case.input["prompt"]}],
            "options": {"temperature": 0.0, "num_predict": 1024},
        }

    def aggregate(self, results: Sequence[CaseResult]) -> PluginAggregate:
        """Score plus richer metrics surfaced in UIs/reports."""
        agg = super().aggregate(results)
        evals = [r.evaluation for r in results]
        metrics: dict[str, Any] = {}
        if evals:
            metrics["pass_at_1"] = round(
                sum(1 for e in evals if e.passed is True) / len(evals), 4
            )
        ms = [e.metrics for e in evals if isinstance(e.metrics, dict)]
        if ms:
            metrics["syntax_ok_ratio"] = _flag_ratio(ms, "syntax_ok")
            metrics["complexity_ok_ratio"] = _flag_ratio(ms, "perf_ok")
            metrics["perf_checked_cases"] = sum(
                1 for m in ms if m.get("perf_checked") is True
            )
            metrics["approach_penalized_cases"] = sum(
                1 for m in ms if m.get("approach_penalty_applied") is True
            )
            metrics["judge_scored_cases"] = sum(
                1 for m in ms if m.get("judge_score") is not None
            )
            passed = sum(
                int(m["tests_passed"]) for m in ms if isinstance(m.get("tests_passed"), int)
            )
            total = sum(
                int(m["tests_total"]) for m in ms if isinstance(m.get("tests_total"), int)
            )
            metrics["tests_passed_total"] = passed
            metrics["tests_total"] = total
            metrics["tests_pass_ratio"] = round(passed / total, 4) if total else None
        agg.metrics = metrics
        return agg

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        expected = case.expected
        fn_name = expected["function_name"]
        harness = expected.get("harness") or ""

        source = extract_python(response.text)
        metrics: dict[str, Any] = {
            "syntax_ok": False,
            "symbol_defined": False,
            "execute_code": bool(ctx.options.get("execute_code", True)),
        }

        if not python_syntax_ok(source):
            metrics["syntax_error"] = True
            return Evaluation(score=0.0, passed=False, metrics=metrics)
        metrics["syntax_ok"] = True

        if not symbol_defined(source, fn_name):
            metrics["symbol_not_found"] = True
            return Evaluation(score=0.0, passed=False, metrics=metrics)
        metrics["symbol_defined"] = True

        enable_perf = bool(ctx.options.get("enable_perf", True))
        timeout = max(1, int(ctx.options.get("timeout_seconds", 30)))
        approach_penalty = float(ctx.options.get("approach_penalty", 0.1))
        judge_weight = float(ctx.options.get("judge_weight", 0.0))

        if not metrics["execute_code"]:
            score = 1.0
            metrics["tests_passed"] = 0
            metrics["tests_total"] = 0
            return await _finalize(
                ctx, case, response, source, score, metrics, approach_penalty, judge_weight
            )

        items = _normalize_items(expected, enable_perf)
        total_weight = sum(item["weight"] for item in items)
        deadline = time.monotonic() + timeout
        passed_weight = 0.0
        stderr_tail = ""
        tests_passed = 0
        tests_total = 0
        perf_checked = 0
        perf_ok_count = 0
        worst_ratio: float | None = None

        for item in items:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            remaining = max(1, remaining)
            if item["kind"] == "perf":
                perf_checked += 1
                ok, ratio = _run_perf_check(
                    harness, source, item, remaining, ctx.options
                )
                if ratio is not None and (worst_ratio is None or ratio > worst_ratio):
                    worst_ratio = ratio
                if ok:
                    perf_ok_count += 1
                    passed_weight += item["weight"]
            else:
                if item["kind"] == "raises":
                    snippet = _raises_snippet(harness, source, item)
                    should_pass = True
                else:
                    snippet = _cmd_snippet(harness, source, item["code"])
                    should_pass = item["should_pass"]
                ok_code, err = _run_test_snippet(snippet, remaining)
                if err:
                    stderr_tail = err[-400:]
                tests_total += 1
                if ok_code == should_pass:
                    tests_passed += 1
                    passed_weight += item["weight"]

        score = passed_weight / total_weight if total_weight else 0.0
        metrics.update(
            {
                "tests_passed": tests_passed,
                "tests_total": tests_total,
                **({"stderr": stderr_tail} if stderr_tail else {}),
            }
        )
        if perf_checked:
            metrics["perf_checked"] = True
            metrics["perf_ok"] = perf_ok_count == perf_checked
            if worst_ratio is not None:
                metrics["worst_perf_ratio"] = round(worst_ratio, 2)
        return await _finalize(
            ctx, case, response, source, score, metrics, approach_penalty, judge_weight
        )


def _normalize_items(expected: dict[str, Any], enable_perf: bool) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for test in expected.get("tests") or []:
        code = test[0]
        should_pass = test[1] if len(test) > 1 else True
        opts = test[2] if len(test) > 2 else {}
        kind = opts.get("kind", "assert")
        item: dict[str, Any] = {
            "kind": kind,
            "code": code,
            "weight": float(opts.get("weight", 1.0)),
            "should_pass": should_pass,
        }
        if kind == "raises":
            item["exc"] = opts.get("exc", "Exception")
        items.append(item)
    if enable_perf:
        for check in expected.get("perf") or []:
            items.append({"kind": "perf", **check, "weight": float(check.get("weight", 1.0))})
    return items


def _cmd_snippet(harness: str, source: str, code: str) -> str:
    return f"{harness}\n{source}\n{code}\n"


def _raises_snippet(harness: str, source: str, item: dict[str, Any]) -> str:
    exc = item["exc"]
    body = "\n".join(
        f"    {line}" if line.strip() else line for line in item["code"].splitlines()
    )
    return (
        f"{harness}\n{source}\n"
        f"try:\n{body}\n"
        f"    raise AssertionError('expected {exc}')\n"
        f"except {exc}:\n"
        f"    pass\n"
    )


async def _finalize(
    ctx,
    case,
    response,
    source: str,
    score: float,
    metrics: dict[str, Any],
    approach_penalty: float,
    judge_weight: float,
) -> Evaluation:
    """Shared tail: static approach penalty + optional judge blend."""
    expected = case.expected
    approach = expected.get("approach")
    if approach:
        checks = approach if isinstance(approach, list) else [approach]
        fired = detect_inefficient(source, checks)
        if fired:
            metrics["approach_flagged"] = fired
            if score >= 1.0 - 1e-9:
                score = max(0.0, score - approach_penalty)
                metrics["approach_penalty_applied"] = True

    score = round(score, 4)
    passed = score >= 1.0 - 1e-9
    evaluation = Evaluation(score=score, passed=passed, metrics=metrics)
    if judge_weight > 0 and getattr(ctx, "judge", None) is not None:
        scored = await judge_evaluation(
            ctx,
            case,
            response,
            rubric=CodingPlugin._JUDGE_RUBRIC,
            deterministic_score=score,
            passed=passed,
            pass_threshold=0.5,
            metrics=metrics,
            judge_weight=judge_weight,
        )
        if scored is not None:
            scored.metrics.setdefault("judge_weight", judge_weight)
            return scored
    return evaluation


def _flag_ratio(metrics: list[dict[str, Any]], key: str) -> float | None:
    num = sum(1 for m in metrics if m.get(key) is True)
    den = sum(1 for m in metrics if key in m)
    return round(num / den, 4) if den else None


def _build_perf_snippet(harness: str, source: str, check: dict[str, Any]) -> str:
    """Harness that times the solution on a small and a large probe.

    Scaling is measured relative (``large_ms / small_ms``) inside one isolated
    subprocess, so machine speed does not matter. Correctness is enforced
    separately by the case's executable assertions; ``eval`` here only ever
    evaluates the case's own literal probe expressions — never model output.
    """
    small = check["small"]
    large = check["large"]
    return f"""\
{harness}
import json
import time

{source}

def _time_once(expr):
    t0 = time.perf_counter()
    eval(expr)
    return (time.perf_counter() - t0) * 1000.0

# warmup each size once, then measure once with the working set loaded
_time_once({small!r})
_time_once({large!r})
s_ms = _time_once({small!r})
l_ms = _time_once({large!r})
print("__PERF__" + json.dumps({{"small_ms": s_ms, "large_ms": l_ms}}))
"""


def _run_perf_check(
    harness: str,
    source: str,
    check: dict[str, Any],
    timeout: int,
    options: dict[str, Any] | None,
) -> tuple[bool, float | None]:
    """Run the scale check. Returns ``(passed, measured_large_over_small)``."""
    ratio = float(check.get("ratio", (options or {}).get("perf_ratio_default", 6.0)))
    snippet = _build_perf_snippet(harness, source, check)
    ok, out = _run_test_snippet(snippet, int(timeout))
    if not ok:
        return False, None
    data = _parse_perf_output(out)
    if not data:
        return False, None
    small_ms = max(float(data["small_ms"]), 1.0)
    large_ms = max(float(data["large_ms"]), 1.0)
    measured = large_ms / small_ms
    return measured <= ratio, round(measured, 2)


def _parse_perf_output(out: str) -> dict[str, Any] | None:
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("__PERF__"):
            try:
                data = json.loads(line[len("__PERF__"):])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _run_test_snippet(snippet: str, timeout: int = 30) -> tuple[bool, str]:
    """Execute a tiny Python snippet in an isolated subprocess.

    Returns ``(ok, stderr_tail)``; ``ok`` is True when the process exits cleanly.
    """
    path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, delete_on_close=False
        ) as f:
            f.write(snippet)
            path = f.name
        proc = subprocess.run(
            # -I = isolated: no user site-packages, no env overrides.
            [sys.executable, "-I", path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        return proc.returncode == 0, (proc.stderr or "") + (proc.stdout or "")
    except (subprocess.TimeoutExpired, OSError, ValueError) as err:
        return False, type(err).__name__
    finally:
        with suppress(Exception):
            os.unlink(path)