export interface Problem {
  id: number;
  slug: string;
  name: string;
  difficulty: "Easy" | "Medium" | "Hard";
  category: string;
  categorySlug: string;
  leetcodeUrl: string;
  neetcodeUrl: string;
  solution: string;
  explanation: string[];
  keyPoints: string[];
  timeComplexity: string;
  spaceComplexity: string;
}

export const problems: Problem[] = [
  // ===== ARRAYS & HASHING =====
  {
    id: 1,
    slug: "contains-duplicate",
    name: "Contains Duplicate",
    difficulty: "Easy",
    category: "Arrays & Hashing",
    categorySlug: "arrays-hashing",
    leetcodeUrl: "https://leetcode.com/problems/contains-duplicate/",
    neetcodeUrl: "https://neetcode.io/problems/duplicate-integer",
    solution: `def containsDuplicate(nums: list[int]) -> bool:
    seen = set()
    for n in nums:
        if n in seen:
            return True
        seen.add(n)
    return False`,
    explanation: [
      "Use a set to track numbers we've seen",
      "For each number, check if it's already in the set",
      "If found, we have a duplicate - return True",
      "Otherwise, add it to the set and continue",
    ],
    keyPoints: [
      "O(n) time, O(n) space",
      "Set provides O(1) lookup",
      "Can also sort and check adjacent elements - O(n log n) time, O(1) space",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(n)",
  },
  {
    id: 2,
    slug: "valid-anagram",
    name: "Valid Anagram",
    difficulty: "Easy",
    category: "Arrays & Hashing",
    categorySlug: "arrays-hashing",
    leetcodeUrl: "https://leetcode.com/problems/valid-anagram/",
    neetcodeUrl: "https://neetcode.io/problems/is-anagram",
    solution: `def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False

    count = {}
    for c in s:
        count[c] = count.get(c, 0) + 1

    for c in t:
        if c not in count:
            return False
        count[c] -= 1
        if count[c] < 0:
            return False

    return True`,
    explanation: [
      "First check if lengths are equal - anagrams must have same length",
      "Count frequency of each character in first string",
      "For each character in second string, decrement count",
      "If any count goes negative or char not found, not an anagram",
    ],
    keyPoints: [
      "O(n) time, O(1) space (26 letters max)",
      "Can also use Counter: Counter(s) == Counter(t)",
      "Sorting approach: sorted(s) == sorted(t) - O(n log n)",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 3,
    slug: "two-sum",
    name: "Two Sum",
    difficulty: "Easy",
    category: "Arrays & Hashing",
    categorySlug: "arrays-hashing",
    leetcodeUrl: "https://leetcode.com/problems/two-sum/",
    neetcodeUrl: "https://neetcode.io/problems/two-integer-sum",
    solution: `def twoSum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, n in enumerate(nums):
        diff = target - n
        if diff in seen:
            return [seen[diff], i]
        seen[n] = i
    return []`,
    explanation: [
      "Use hashmap to store each number and its index",
      "For each number, calculate the complement (target - num)",
      "Check if complement exists in hashmap",
      "If found, return both indices; otherwise store current number",
    ],
    keyPoints: [
      "O(n) time, O(n) space",
      "Hashmap for O(1) lookup of complement",
      "Store index, not just the value",
      "One-pass solution - check and add in same loop",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(n)",
  },
  {
    id: 4,
    slug: "group-anagrams",
    name: "Group Anagrams",
    difficulty: "Medium",
    category: "Arrays & Hashing",
    categorySlug: "arrays-hashing",
    leetcodeUrl: "https://leetcode.com/problems/group-anagrams/",
    neetcodeUrl: "https://neetcode.io/problems/anagram-groups",
    solution: `def groupAnagrams(strs: list[str]) -> list[list[str]]:
    groups = {}
    for s in strs:
        key = tuple(sorted(s))
        if key not in groups:
            groups[key] = []
        groups[key].append(s)
    return list(groups.values())`,
    explanation: [
      "Anagrams have the same sorted characters",
      "Use sorted string (as tuple) as the key",
      "Group all strings with the same key together",
      "Return all groups as a list of lists",
    ],
    keyPoints: [
      "O(n * k log k) where k is max string length",
      "Alternative: use character count tuple as key - O(n * k)",
      "Key insight: anagrams share the same 'signature'",
    ],
    timeComplexity: "O(n * k log k)",
    spaceComplexity: "O(n * k)",
  },
  {
    id: 5,
    slug: "top-k-frequent-elements",
    name: "Top K Frequent Elements",
    difficulty: "Medium",
    category: "Arrays & Hashing",
    categorySlug: "arrays-hashing",
    leetcodeUrl: "https://leetcode.com/problems/top-k-frequent-elements/",
    neetcodeUrl: "https://neetcode.io/problems/top-k-elements-in-list",
    solution: `def topKFrequent(nums: list[int], k: int) -> list[int]:
    count = {}
    for n in nums:
        count[n] = count.get(n, 0) + 1

    # Bucket sort by frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)

    result = []
    for i in range(len(buckets) - 1, -1, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    return result`,
    explanation: [
      "First count frequency of each number",
      "Use bucket sort: index = frequency, value = list of numbers",
      "Iterate from highest frequency bucket to lowest",
      "Collect numbers until we have k elements",
    ],
    keyPoints: [
      "O(n) time with bucket sort approach",
      "Heap approach: O(n log k)",
      "Max frequency is at most n (array length)",
      "Bucket sort avoids sorting overhead",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(n)",
  },
  {
    id: 6,
    slug: "valid-sudoku",
    name: "Valid Sudoku",
    difficulty: "Medium",
    category: "Arrays & Hashing",
    categorySlug: "arrays-hashing",
    leetcodeUrl: "https://leetcode.com/problems/valid-sudoku/",
    neetcodeUrl: "https://neetcode.io/problems/valid-sudoku",
    solution: `def isValidSudoku(board: list[list[str]]) -> bool:
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]

    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == '.':
                continue

            box_idx = (r // 3) * 3 + (c // 3)

            if val in rows[r] or val in cols[c] or val in boxes[box_idx]:
                return False

            rows[r].add(val)
            cols[c].add(val)
            boxes[box_idx].add(val)

    return True`,
    explanation: [
      "Track seen numbers for each row, column, and 3x3 box",
      "Use sets for O(1) lookup of duplicates",
      "Calculate box index: (row // 3) * 3 + (col // 3)",
      "If number already exists in any set, board is invalid",
    ],
    keyPoints: [
      "O(81) = O(1) time and space (fixed board size)",
      "Box index formula maps 9x9 grid to 9 boxes",
      "Skip empty cells marked with '.'",
    ],
    timeComplexity: "O(1)",
    spaceComplexity: "O(1)",
  },
  {
    id: 7,
    slug: "product-of-array-except-self",
    name: "Product of Array Except Self",
    difficulty: "Medium",
    category: "Arrays & Hashing",
    categorySlug: "arrays-hashing",
    leetcodeUrl: "https://leetcode.com/problems/product-of-array-except-self/",
    neetcodeUrl: "https://neetcode.io/problems/products-of-array-discluding-self",
    solution: `def productExceptSelf(nums: list[int]) -> list[int]:
    n = len(nums)
    result = [1] * n

    # Left products
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]

    # Right products
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]

    return result`,
    explanation: [
      "For each position, we need product of all elements except itself",
      "This equals (product of all left elements) × (product of all right elements)",
      "First pass: store prefix products (everything to the left)",
      "Second pass: multiply by suffix products (everything to the right)",
    ],
    keyPoints: [
      "O(n) time, O(1) extra space (output doesn't count)",
      "Cannot use division (constraint)",
      "Two-pass approach: prefix then suffix",
      "Key insight: result[i] = prefix[i] * suffix[i]",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 8,
    slug: "longest-consecutive-sequence",
    name: "Longest Consecutive Sequence",
    difficulty: "Medium",
    category: "Arrays & Hashing",
    categorySlug: "arrays-hashing",
    leetcodeUrl: "https://leetcode.com/problems/longest-consecutive-sequence/",
    neetcodeUrl: "https://neetcode.io/problems/longest-consecutive-sequence",
    solution: `def longestConsecutive(nums: list[int]) -> int:
    num_set = set(nums)
    longest = 0

    for n in num_set:
        # Only start counting from sequence beginning
        if n - 1 not in num_set:
            length = 1
            while n + length in num_set:
                length += 1
            longest = max(longest, length)

    return longest`,
    explanation: [
      "Convert to set for O(1) lookups",
      "Only start counting from the beginning of a sequence",
      "A number is a sequence start if (n-1) is not in set",
      "Count consecutive numbers from each starting point",
    ],
    keyPoints: [
      "O(n) time despite nested loop - each number visited at most twice",
      "O(n) space for the set",
      "Key optimization: only start from sequence beginnings",
      "Without optimization, would be O(n²)",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(n)",
  },

  // ===== TWO POINTERS =====
  {
    id: 9,
    slug: "valid-palindrome",
    name: "Valid Palindrome",
    difficulty: "Easy",
    category: "Two Pointers",
    categorySlug: "two-pointers",
    leetcodeUrl: "https://leetcode.com/problems/valid-palindrome/",
    neetcodeUrl: "https://neetcode.io/problems/is-palindrome",
    solution: `def isPalindrome(s: str) -> bool:
    left, right = 0, len(s) - 1

    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True`,
    explanation: [
      "Use two pointers starting from both ends",
      "Skip non-alphanumeric characters",
      "Compare characters (case-insensitive)",
      "If any mismatch, not a palindrome",
    ],
    keyPoints: [
      "O(n) time, O(1) space",
      "isalnum() checks for letters and digits",
      "lower() for case-insensitive comparison",
      "Alternative: clean string first, then compare s == s[::-1]",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 10,
    slug: "two-sum-ii",
    name: "Two Sum II - Input Array Is Sorted",
    difficulty: "Medium",
    category: "Two Pointers",
    categorySlug: "two-pointers",
    leetcodeUrl: "https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/",
    neetcodeUrl: "https://neetcode.io/problems/two-integer-sum-ii",
    solution: `def twoSum(numbers: list[int], target: int) -> list[int]:
    left, right = 0, len(numbers) - 1

    while left < right:
        curr_sum = numbers[left] + numbers[right]

        if curr_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif curr_sum < target:
            left += 1
        else:
            right -= 1

    return []`,
    explanation: [
      "Array is sorted - use two pointers from both ends",
      "If sum < target, move left pointer right (need larger)",
      "If sum > target, move right pointer left (need smaller)",
      "If sum == target, found the pair",
    ],
    keyPoints: [
      "O(n) time, O(1) space",
      "Only works because array is sorted",
      "Returns 1-indexed positions",
      "Guaranteed exactly one solution exists",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 11,
    slug: "3sum",
    name: "3Sum",
    difficulty: "Medium",
    category: "Two Pointers",
    categorySlug: "two-pointers",
    leetcodeUrl: "https://leetcode.com/problems/3sum/",
    neetcodeUrl: "https://neetcode.io/problems/three-integer-sum",
    solution: `def threeSum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result`,
    explanation: [
      "Sort the array first",
      "Fix first number, then use two-pointer for remaining two",
      "Skip duplicates at all three positions to avoid duplicate triplets",
      "Move pointers based on whether sum is too small or too large",
    ],
    keyPoints: [
      "O(n²) time, O(1) or O(n) space depending on sort",
      "Sorting enables two-pointer technique",
      "Must handle duplicates carefully",
      "Reduces to Two Sum II for each fixed first element",
    ],
    timeComplexity: "O(n²)",
    spaceComplexity: "O(1)",
  },
  {
    id: 12,
    slug: "container-with-most-water",
    name: "Container With Most Water",
    difficulty: "Medium",
    category: "Two Pointers",
    categorySlug: "two-pointers",
    leetcodeUrl: "https://leetcode.com/problems/container-with-most-water/",
    neetcodeUrl: "https://neetcode.io/problems/max-water-container",
    solution: `def maxArea(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    max_area = 0

    while left < right:
        width = right - left
        h = min(height[left], height[right])
        max_area = max(max_area, width * h)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area`,
    explanation: [
      "Start with widest possible container (both ends)",
      "Area = width × min(left_height, right_height)",
      "Move the pointer with smaller height inward",
      "Moving smaller height might find taller line; moving taller won't help",
    ],
    keyPoints: [
      "O(n) time, O(1) space",
      "Greedy: always move the shorter line",
      "Width decreases, so we need taller lines to improve",
      "Moving taller line can only decrease or maintain area",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },

  // ===== SLIDING WINDOW =====
  {
    id: 13,
    slug: "best-time-to-buy-and-sell-stock",
    name: "Best Time to Buy and Sell Stock",
    difficulty: "Easy",
    category: "Sliding Window",
    categorySlug: "sliding-window",
    leetcodeUrl: "https://leetcode.com/problems/best-time-to-buy-and-sell-stock/",
    neetcodeUrl: "https://neetcode.io/problems/buy-and-sell-crypto",
    solution: `def maxProfit(prices: list[int]) -> int:
    min_price = float('inf')
    max_profit = 0

    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)

    return max_profit`,
    explanation: [
      "Track the minimum price seen so far",
      "At each day, calculate profit if we sold today",
      "Update max profit if current profit is higher",
      "One pass through the array",
    ],
    keyPoints: [
      "O(n) time, O(1) space",
      "Must buy before selling (order matters)",
      "Track running minimum and maximum profit",
      "Can also think of as sliding window with left at min price",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 14,
    slug: "longest-substring-without-repeating-characters",
    name: "Longest Substring Without Repeating Characters",
    difficulty: "Medium",
    category: "Sliding Window",
    categorySlug: "sliding-window",
    leetcodeUrl: "https://leetcode.com/problems/longest-substring-without-repeating-characters/",
    neetcodeUrl: "https://neetcode.io/problems/longest-substring-without-duplicates",
    solution: `def lengthOfLongestSubstring(s: str) -> int:
    char_index = {}
    max_len = 0
    left = 0

    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1

        char_index[char] = right
        max_len = max(max_len, right - left + 1)

    return max_len`,
    explanation: [
      "Use sliding window with hashmap to track character indices",
      "Expand window by moving right pointer",
      "When duplicate found, shrink window by moving left past the duplicate",
      "Track maximum window size throughout",
    ],
    keyPoints: [
      "O(n) time, O(min(n, alphabet)) space",
      "Window contains only unique characters",
      "Store last index of each character",
      "Only move left if duplicate is within current window",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(min(n, m))",
  },
  {
    id: 15,
    slug: "longest-repeating-character-replacement",
    name: "Longest Repeating Character Replacement",
    difficulty: "Medium",
    category: "Sliding Window",
    categorySlug: "sliding-window",
    leetcodeUrl: "https://leetcode.com/problems/longest-repeating-character-replacement/",
    neetcodeUrl: "https://neetcode.io/problems/longest-repeating-substring-with-replacement",
    solution: `def characterReplacement(s: str, k: int) -> int:
    count = {}
    max_freq = 0
    left = 0
    max_len = 0

    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1
        max_freq = max(max_freq, count[s[right]])

        # Window size - max frequency = chars to replace
        while (right - left + 1) - max_freq > k:
            count[s[left]] -= 1
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len`,
    explanation: [
      "Sliding window: window is valid if we can make all chars same with ≤k replacements",
      "Window size - max frequency = number of chars that need replacing",
      "If this exceeds k, shrink window from left",
      "Track the maximum valid window size",
    ],
    keyPoints: [
      "O(n) time, O(26) = O(1) space",
      "Key insight: window_size - max_freq ≤ k",
      "Don't need to decrease max_freq when shrinking (optimization)",
      "max_freq only needs to increase for answer to improve",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },

  // ===== STACK =====
  {
    id: 16,
    slug: "valid-parentheses",
    name: "Valid Parentheses",
    difficulty: "Easy",
    category: "Stack",
    categorySlug: "stack",
    leetcodeUrl: "https://leetcode.com/problems/valid-parentheses/",
    neetcodeUrl: "https://neetcode.io/problems/validate-parentheses",
    solution: `def isValid(s: str) -> bool:
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
        else:
            stack.append(char)

    return len(stack) == 0`,
    explanation: [
      "Push opening brackets onto stack",
      "For closing brackets, check if top of stack matches",
      "If match, pop; if not, invalid",
      "At end, stack should be empty",
    ],
    keyPoints: [
      "O(n) time, O(n) space",
      "Stack tracks unmatched opening brackets",
      "Map closing to opening for easy lookup",
      "Empty stack at end means all brackets matched",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(n)",
  },
  {
    id: 17,
    slug: "min-stack",
    name: "Min Stack",
    difficulty: "Medium",
    category: "Stack",
    categorySlug: "stack",
    leetcodeUrl: "https://leetcode.com/problems/min-stack/",
    neetcodeUrl: "https://neetcode.io/problems/minimum-stack",
    solution: `class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        if self.stack[-1] == self.min_stack[-1]:
            self.min_stack.pop()
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]`,
    explanation: [
      "Use two stacks: main stack and min stack",
      "Min stack tracks minimum at each level",
      "Push to min stack only when value ≤ current min",
      "Pop from min stack when popping the current minimum",
    ],
    keyPoints: [
      "O(1) for all operations",
      "O(n) space for both stacks",
      "Min stack stores running minimums",
      "Alternative: store (value, min_so_far) pairs",
    ],
    timeComplexity: "O(1)",
    spaceComplexity: "O(n)",
  },
  {
    id: 18,
    slug: "daily-temperatures",
    name: "Daily Temperatures",
    difficulty: "Medium",
    category: "Stack",
    categorySlug: "stack",
    leetcodeUrl: "https://leetcode.com/problems/daily-temperatures/",
    neetcodeUrl: "https://neetcode.io/problems/daily-temperatures",
    solution: `def dailyTemperatures(temperatures: list[int]) -> list[int]:
    n = len(temperatures)
    result = [0] * n
    stack = []  # Stack of indices

    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_idx = stack.pop()
            result[prev_idx] = i - prev_idx
        stack.append(i)

    return result`,
    explanation: [
      "Use monotonic decreasing stack (store indices)",
      "For each temperature, pop all smaller temperatures from stack",
      "For each popped index, calculate days waited",
      "Push current index onto stack",
    ],
    keyPoints: [
      "O(n) time, O(n) space",
      "Monotonic stack pattern: finding next greater element",
      "Stack stores indices of temperatures waiting for warmer day",
      "Each index pushed and popped at most once",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(n)",
  },

  // ===== BINARY SEARCH =====
  {
    id: 19,
    slug: "binary-search",
    name: "Binary Search",
    difficulty: "Easy",
    category: "Binary Search",
    categorySlug: "binary-search",
    leetcodeUrl: "https://leetcode.com/problems/binary-search/",
    neetcodeUrl: "https://neetcode.io/problems/binary-search",
    solution: `def search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1`,
    explanation: [
      "Search in sorted array by repeatedly halving search space",
      "Compare middle element with target",
      "If equal, found; if less, search right half; if more, search left half",
      "Continue until found or search space exhausted",
    ],
    keyPoints: [
      "O(log n) time, O(1) space",
      "Use left + (right - left) // 2 to avoid overflow",
      "Condition: left <= right (inclusive bounds)",
      "Foundation for many binary search variants",
    ],
    timeComplexity: "O(log n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 20,
    slug: "find-minimum-in-rotated-sorted-array",
    name: "Find Minimum in Rotated Sorted Array",
    difficulty: "Medium",
    category: "Binary Search",
    categorySlug: "binary-search",
    leetcodeUrl: "https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/",
    neetcodeUrl: "https://neetcode.io/problems/find-minimum-in-rotated-sorted-array",
    solution: `def findMin(nums: list[int]) -> int:
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] > nums[right]:
            # Minimum is in right half
            left = mid + 1
        else:
            # Minimum is in left half (including mid)
            right = mid

    return nums[left]`,
    explanation: [
      "Rotated sorted array has two sorted portions",
      "Compare mid with right to determine which half has minimum",
      "If mid > right, minimum is in right half",
      "If mid ≤ right, minimum is in left half (including mid)",
    ],
    keyPoints: [
      "O(log n) time, O(1) space",
      "Compare with right element, not left",
      "Minimum is at the rotation pivot",
      "Use left < right (not <=) since we're finding minimum",
    ],
    timeComplexity: "O(log n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 21,
    slug: "search-in-rotated-sorted-array",
    name: "Search in Rotated Sorted Array",
    difficulty: "Medium",
    category: "Binary Search",
    categorySlug: "binary-search",
    leetcodeUrl: "https://leetcode.com/problems/search-in-rotated-sorted-array/",
    neetcodeUrl: "https://neetcode.io/problems/search-in-rotated-sorted-array",
    solution: `def search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1`,
    explanation: [
      "At least one half of the array is always sorted",
      "Determine which half is sorted by comparing left with mid",
      "Check if target is in the sorted half",
      "If yes, search that half; otherwise, search the other half",
    ],
    keyPoints: [
      "O(log n) time, O(1) space",
      "Key insight: one half is always sorted",
      "Check if target is in sorted range",
      "Combine finding sorted half + binary search",
    ],
    timeComplexity: "O(log n)",
    spaceComplexity: "O(1)",
  },

  // ===== LINKED LIST =====
  {
    id: 22,
    slug: "reverse-linked-list",
    name: "Reverse Linked List",
    difficulty: "Easy",
    category: "Linked List",
    categorySlug: "linked-list",
    leetcodeUrl: "https://leetcode.com/problems/reverse-linked-list/",
    neetcodeUrl: "https://neetcode.io/problems/reverse-a-linked-list",
    solution: `def reverseList(head: ListNode) -> ListNode:
    prev = None
    curr = head

    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp

    return prev`,
    explanation: [
      "Use three pointers: prev, curr, next",
      "For each node, reverse its next pointer to point to previous",
      "Save next node before changing pointer",
      "Move all pointers forward by one",
    ],
    keyPoints: [
      "O(n) time, O(1) space",
      "Iterative approach shown (can also do recursive)",
      "prev starts as None (new tail)",
      "Return prev (new head) at end",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 23,
    slug: "merge-two-sorted-lists",
    name: "Merge Two Sorted Lists",
    difficulty: "Easy",
    category: "Linked List",
    categorySlug: "linked-list",
    leetcodeUrl: "https://leetcode.com/problems/merge-two-sorted-lists/",
    neetcodeUrl: "https://neetcode.io/problems/merge-two-sorted-linked-lists",
    solution: `def mergeTwoLists(list1: ListNode, list2: ListNode) -> ListNode:
    dummy = ListNode(0)
    curr = dummy

    while list1 and list2:
        if list1.val <= list2.val:
            curr.next = list1
            list1 = list1.next
        else:
            curr.next = list2
            list2 = list2.next
        curr = curr.next

    curr.next = list1 or list2
    return dummy.next`,
    explanation: [
      "Use dummy node to simplify edge cases",
      "Compare heads of both lists, take smaller one",
      "Advance pointer in list we took from",
      "At end, attach remaining nodes from non-empty list",
    ],
    keyPoints: [
      "O(n + m) time, O(1) space",
      "Dummy node pattern avoids special case for head",
      "Return dummy.next as actual head",
      "list1 or list2 handles remaining elements",
    ],
    timeComplexity: "O(n + m)",
    spaceComplexity: "O(1)",
  },
  {
    id: 24,
    slug: "reorder-list",
    name: "Reorder List",
    difficulty: "Medium",
    category: "Linked List",
    categorySlug: "linked-list",
    leetcodeUrl: "https://leetcode.com/problems/reorder-list/",
    neetcodeUrl: "https://neetcode.io/problems/reorder-linked-list",
    solution: `def reorderList(head: ListNode) -> None:
    if not head or not head.next:
        return

    # Find middle
    slow, fast = head, head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # Reverse second half
    prev, curr = None, slow.next
    slow.next = None
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp

    # Merge two halves
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first, second = tmp1, tmp2`,
    explanation: [
      "Find middle using slow/fast pointers",
      "Reverse the second half of the list",
      "Merge first half and reversed second half alternately",
      "Pattern: L0→Ln→L1→Ln-1→L2→Ln-2→...",
    ],
    keyPoints: [
      "O(n) time, O(1) space",
      "Combines three patterns: find middle, reverse, merge",
      "In-place modification, no new nodes created",
      "Must handle odd/even length lists",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 25,
    slug: "remove-nth-node-from-end-of-list",
    name: "Remove Nth Node From End of List",
    difficulty: "Medium",
    category: "Linked List",
    categorySlug: "linked-list",
    leetcodeUrl: "https://leetcode.com/problems/remove-nth-node-from-end-of-list/",
    neetcodeUrl: "https://neetcode.io/problems/remove-node-from-end-of-linked-list",
    solution: `def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    dummy = ListNode(0, head)
    slow, fast = dummy, dummy

    # Move fast n+1 steps ahead
    for _ in range(n + 1):
        fast = fast.next

    # Move both until fast reaches end
    while fast:
        slow = slow.next
        fast = fast.next

    # Remove the node
    slow.next = slow.next.next

    return dummy.next`,
    explanation: [
      "Use two pointers with n+1 gap",
      "When fast reaches end, slow is at node before target",
      "Delete by skipping: slow.next = slow.next.next",
      "Dummy node handles edge case of removing head",
    ],
    keyPoints: [
      "O(n) time, O(1) space - single pass",
      "Gap of n+1 so slow stops one before target",
      "Dummy node simplifies removing first node",
      "Could also do two passes: count length first",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 26,
    slug: "linked-list-cycle",
    name: "Linked List Cycle",
    difficulty: "Easy",
    category: "Linked List",
    categorySlug: "linked-list",
    leetcodeUrl: "https://leetcode.com/problems/linked-list-cycle/",
    neetcodeUrl: "https://neetcode.io/problems/linked-list-cycle-detection",
    solution: `def hasCycle(head: ListNode) -> bool:
    slow, fast = head, head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            return True

    return False`,
    explanation: [
      "Floyd's Cycle Detection (tortoise and hare)",
      "Slow pointer moves one step, fast moves two",
      "If cycle exists, fast will eventually catch slow",
      "If no cycle, fast reaches end (None)",
    ],
    keyPoints: [
      "O(n) time, O(1) space",
      "Fast catches slow within one cycle length",
      "Check fast and fast.next before advancing",
      "Can extend to find cycle start (move one pointer to head, both move 1 step)",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 27,
    slug: "lru-cache",
    name: "LRU Cache",
    difficulty: "Medium",
    category: "Linked List",
    categorySlug: "linked-list",
    leetcodeUrl: "https://leetcode.com/problems/lru-cache/",
    neetcodeUrl: "https://neetcode.io/problems/lru-cache",
    solution: `class Node:
    def __init__(self, key=0, val=0):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node
        self.head = Node()  # dummy head (most recent)
        self.tail = Node()  # dummy tail (least recent)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_to_front(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self.cache[key] = node
        self._add_to_front(node)
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]`,
    explanation: [
      "Hashmap for O(1) key lookup, doubly linked list for O(1) ordering",
      "Most recently used at head, least recently used at tail",
      "On access: move node to front",
      "On insert: add to front, evict from tail if over capacity",
    ],
    keyPoints: [
      "O(1) for both get and put operations",
      "Doubly linked list enables O(1) removal and insertion",
      "Dummy head/tail simplify edge cases",
      "Store key in node to find cache entry during eviction",
    ],
    timeComplexity: "O(1)",
    spaceComplexity: "O(capacity)",
  },

  // ===== TREES =====
  {
    id: 28,
    slug: "invert-binary-tree",
    name: "Invert Binary Tree",
    difficulty: "Easy",
    category: "Trees",
    categorySlug: "trees",
    leetcodeUrl: "https://leetcode.com/problems/invert-binary-tree/",
    neetcodeUrl: "https://neetcode.io/problems/invert-a-binary-tree",
    solution: `def invertTree(root: TreeNode) -> TreeNode:
    if not root:
        return None

    root.left, root.right = root.right, root.left

    invertTree(root.left)
    invertTree(root.right)

    return root`,
    explanation: [
      "Swap left and right children at each node",
      "Recursively invert left and right subtrees",
      "Order doesn't matter: swap before or after recursion",
      "Base case: return None for empty tree",
    ],
    keyPoints: [
      "O(n) time, O(h) space where h = height",
      "Simple recursive solution",
      "Can also use BFS/DFS iteratively",
      "Famous problem: Homebrew creator couldn't solve on whiteboard",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(h)",
  },
  {
    id: 29,
    slug: "maximum-depth-of-binary-tree",
    name: "Maximum Depth of Binary Tree",
    difficulty: "Easy",
    category: "Trees",
    categorySlug: "trees",
    leetcodeUrl: "https://leetcode.com/problems/maximum-depth-of-binary-tree/",
    neetcodeUrl: "https://neetcode.io/problems/depth-of-binary-tree",
    solution: `def maxDepth(root: TreeNode) -> int:
    if not root:
        return 0

    return 1 + max(maxDepth(root.left), maxDepth(root.right))`,
    explanation: [
      "Depth = 1 + max(left depth, right depth)",
      "Base case: empty tree has depth 0",
      "Recursively find depth of each subtree",
      "Add 1 for current node",
    ],
    keyPoints: [
      "O(n) time, O(h) space",
      "Classic tree recursion pattern",
      "Can also use BFS (level order) - count levels",
      "Iterative DFS with stack also works",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(h)",
  },
  {
    id: 30,
    slug: "diameter-of-binary-tree",
    name: "Diameter of Binary Tree",
    difficulty: "Easy",
    category: "Trees",
    categorySlug: "trees",
    leetcodeUrl: "https://leetcode.com/problems/diameter-of-binary-tree/",
    neetcodeUrl: "https://neetcode.io/problems/binary-tree-diameter",
    solution: `def diameterOfBinaryTree(root: TreeNode) -> int:
    diameter = 0

    def height(node):
        nonlocal diameter
        if not node:
            return 0

        left_h = height(node.left)
        right_h = height(node.right)

        diameter = max(diameter, left_h + right_h)

        return 1 + max(left_h, right_h)

    height(root)
    return diameter`,
    explanation: [
      "Diameter through a node = left height + right height",
      "Track maximum diameter seen across all nodes",
      "Return height at each node for parent's calculation",
      "Diameter may not pass through root",
    ],
    keyPoints: [
      "O(n) time, O(h) space",
      "Diameter is longest path between any two nodes",
      "Path = number of edges (not nodes)",
      "Combine height calculation with diameter tracking",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(h)",
  },
  {
    id: 31,
    slug: "balanced-binary-tree",
    name: "Balanced Binary Tree",
    difficulty: "Easy",
    category: "Trees",
    categorySlug: "trees",
    leetcodeUrl: "https://leetcode.com/problems/balanced-binary-tree/",
    neetcodeUrl: "https://neetcode.io/problems/balanced-binary-tree",
    solution: `def isBalanced(root: TreeNode) -> bool:
    def check(node):
        if not node:
            return 0

        left = check(node.left)
        right = check(node.right)

        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1

        return 1 + max(left, right)

    return check(root) != -1`,
    explanation: [
      "Balanced = height difference ≤ 1 at every node",
      "Return -1 to signal imbalance (early termination)",
      "Otherwise return height for parent's check",
      "Check both subtrees are balanced AND heights differ by ≤ 1",
    ],
    keyPoints: [
      "O(n) time, O(h) space",
      "Use -1 as sentinel for imbalanced subtree",
      "Single pass: combine balance check with height calculation",
      "Naive approach (check height at each node) is O(n²)",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(h)",
  },
  {
    id: 32,
    slug: "same-tree",
    name: "Same Tree",
    difficulty: "Easy",
    category: "Trees",
    categorySlug: "trees",
    leetcodeUrl: "https://leetcode.com/problems/same-tree/",
    neetcodeUrl: "https://neetcode.io/problems/same-binary-tree",
    solution: `def isSameTree(p: TreeNode, q: TreeNode) -> bool:
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.val != q.val:
        return False

    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)`,
    explanation: [
      "Base case: both None = same, one None = different",
      "Compare values at current nodes",
      "Recursively check left and right subtrees match",
      "All conditions must be true for trees to be same",
    ],
    keyPoints: [
      "O(n) time, O(h) space",
      "Three base cases: both null, one null, values differ",
      "Can also use iterative BFS with two queues",
      "Foundation for subtree problem",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(h)",
  },
  {
    id: 33,
    slug: "subtree-of-another-tree",
    name: "Subtree of Another Tree",
    difficulty: "Easy",
    category: "Trees",
    categorySlug: "trees",
    leetcodeUrl: "https://leetcode.com/problems/subtree-of-another-tree/",
    neetcodeUrl: "https://neetcode.io/problems/subtree-of-a-binary-tree",
    solution: `def isSubtree(root: TreeNode, subRoot: TreeNode) -> bool:
    def isSame(p, q):
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return isSame(p.left, q.left) and isSame(p.right, q.right)

    if not root:
        return False

    if isSame(root, subRoot):
        return True

    return isSubtree(root.left, subRoot) or isSubtree(root.right, subRoot)`,
    explanation: [
      "At each node in root, check if subtree matches subRoot",
      "Use isSameTree helper to compare trees",
      "Recursively check all nodes as potential subtree roots",
      "Return True if any node matches",
    ],
    keyPoints: [
      "O(n * m) time where n = root size, m = subRoot size",
      "O(h) space for recursion",
      "Can optimize with tree hashing or KMP-like approach",
      "Empty tree is subtree of any tree",
    ],
    timeComplexity: "O(n * m)",
    spaceComplexity: "O(h)",
  },
  {
    id: 34,
    slug: "lowest-common-ancestor-of-a-binary-search-tree",
    name: "Lowest Common Ancestor of a Binary Search Tree",
    difficulty: "Medium",
    category: "Trees",
    categorySlug: "trees",
    leetcodeUrl: "https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/",
    neetcodeUrl: "https://neetcode.io/problems/lowest-common-ancestor-in-binary-search-tree",
    solution: `def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
    return None`,
    explanation: [
      "Use BST property: left < root < right",
      "If both p and q are smaller, LCA is in left subtree",
      "If both are larger, LCA is in right subtree",
      "If split (one on each side), current node is LCA",
    ],
    keyPoints: [
      "O(h) time, O(1) space (iterative)",
      "Exploits BST ordering property",
      "LCA is where p and q split to different subtrees",
      "Works even if p or q equals root",
    ],
    timeComplexity: "O(h)",
    spaceComplexity: "O(1)",
  },
  {
    id: 35,
    slug: "binary-tree-level-order-traversal",
    name: "Binary Tree Level Order Traversal",
    difficulty: "Medium",
    category: "Trees",
    categorySlug: "trees",
    leetcodeUrl: "https://leetcode.com/problems/binary-tree-level-order-traversal/",
    neetcodeUrl: "https://neetcode.io/problems/level-order-traversal-of-binary-tree",
    solution: `from collections import deque

def levelOrder(root: TreeNode) -> list[list[int]]:
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        level_size = len(queue)

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result`,
    explanation: [
      "Use BFS with queue to traverse level by level",
      "Track level size at start of each level",
      "Process exactly level_size nodes for current level",
      "Add children to queue for next level",
    ],
    keyPoints: [
      "O(n) time, O(w) space where w = max width",
      "Classic BFS pattern for trees",
      "Process one level at a time",
      "Foundation for many level-based problems",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(w)",
  },
  {
    id: 36,
    slug: "binary-tree-right-side-view",
    name: "Binary Tree Right Side View",
    difficulty: "Medium",
    category: "Trees",
    categorySlug: "trees",
    leetcodeUrl: "https://leetcode.com/problems/binary-tree-right-side-view/",
    neetcodeUrl: "https://neetcode.io/problems/binary-tree-right-side-view",
    solution: `from collections import deque

def rightSideView(root: TreeNode) -> list[int]:
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)

        for i in range(level_size):
            node = queue.popleft()

            if i == level_size - 1:
                result.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return result`,
    explanation: [
      "Level order traversal, but only keep last node of each level",
      "Last node at each level is visible from right side",
      "Could also do DFS: visit right child first, track depth",
      "BFS approach is more intuitive",
    ],
    keyPoints: [
      "O(n) time, O(w) space",
      "Variant of level order traversal",
      "Take rightmost node at each level",
      "DFS alternative: maintain result[depth] = rightmost at that depth",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(w)",
  },
  {
    id: 37,
    slug: "count-good-nodes-in-binary-tree",
    name: "Count Good Nodes in Binary Tree",
    difficulty: "Medium",
    category: "Trees",
    categorySlug: "trees",
    leetcodeUrl: "https://leetcode.com/problems/count-good-nodes-in-binary-tree/",
    neetcodeUrl: "https://neetcode.io/problems/count-good-nodes-in-binary-tree",
    solution: `def goodNodes(root: TreeNode) -> int:
    def dfs(node, max_so_far):
        if not node:
            return 0

        count = 1 if node.val >= max_so_far else 0
        new_max = max(max_so_far, node.val)

        count += dfs(node.left, new_max)
        count += dfs(node.right, new_max)

        return count

    return dfs(root, root.val)`,
    explanation: [
      "Good node: no node with greater value on path from root",
      "Track maximum value seen on path from root",
      "Node is good if its value >= max seen so far",
      "Pass updated max to children",
    ],
    keyPoints: [
      "O(n) time, O(h) space",
      "DFS while tracking path maximum",
      "Root is always a good node",
      "Update max when going deeper, don't decrease it",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(h)",
  },
  {
    id: 38,
    slug: "validate-binary-search-tree",
    name: "Validate Binary Search Tree",
    difficulty: "Medium",
    category: "Trees",
    categorySlug: "trees",
    leetcodeUrl: "https://leetcode.com/problems/validate-binary-search-tree/",
    neetcodeUrl: "https://neetcode.io/problems/valid-binary-search-tree",
    solution: `def isValidBST(root: TreeNode) -> bool:
    def validate(node, min_val, max_val):
        if not node:
            return True

        if node.val <= min_val or node.val >= max_val:
            return False

        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))

    return validate(root, float('-inf'), float('inf'))`,
    explanation: [
      "Track valid range for each node",
      "Left child must be < parent; right child must be > parent",
      "Update bounds when going left (new max) or right (new min)",
      "Node must be strictly within (min, max) range",
    ],
    keyPoints: [
      "O(n) time, O(h) space",
      "Pass min/max bounds down the tree",
      "Common mistake: only comparing with parent",
      "Alternative: inorder traversal should be strictly increasing",
    ],
    timeComplexity: "O(n)",
    spaceComplexity: "O(h)",
  },
  {
    id: 39,
    slug: "kth-smallest-element-in-a-bst",
    name: "Kth Smallest Element in a BST",
    difficulty: "Medium",
    category: "Trees",
    categorySlug: "trees",
    leetcodeUrl: "https://leetcode.com/problems/kth-smallest-element-in-a-bst/",
    neetcodeUrl: "https://neetcode.io/problems/kth-smallest-integer-in-bst",
    solution: `def kthSmallest(root: TreeNode, k: int) -> int:
    stack = []
    curr = root
    count = 0

    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left

        curr = stack.pop()
        count += 1

        if count == k:
            return curr.val

        curr = curr.right

    return -1`,
    explanation: [
      "Inorder traversal of BST visits nodes in sorted order",
      "Use iterative approach with stack",
      "Go left as far as possible, then visit, then go right",
      "Count nodes visited; return when count equals k",
    ],
    keyPoints: [
      "O(h + k) time, O(h) space",
      "Inorder: left → root → right gives sorted order",
      "Iterative avoids processing all nodes if k is small",
      "Could also do recursive but harder to stop early",
    ],
    timeComplexity: "O(h + k)",
    spaceComplexity: "O(h)",
  },

  // ===== HEAP =====
  {
    id: 40,
    slug: "kth-largest-element-in-a-stream",
    name: "Kth Largest Element in a Stream",
    difficulty: "Easy",
    category: "Heap",
    categorySlug: "heap",
    leetcodeUrl: "https://leetcode.com/problems/kth-largest-element-in-a-stream/",
    neetcodeUrl: "https://neetcode.io/problems/kth-largest-integer-in-a-stream",
    solution: `import heapq

class KthLargest:
    def __init__(self, k: int, nums: list[int]):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)

        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)

        if len(self.heap) > self.k:
            heapq.heappop(self.heap)

        return self.heap[0]`,
    explanation: [
      "Maintain a min-heap of size k",
      "Min-heap root is the kth largest overall",
      "On add: push to heap, then pop if size > k",
      "Popping smallest keeps k largest elements",
    ],
    keyPoints: [
      "O(log k) for add, O(n log k) for init",
      "O(k) space for heap",
      "Min-heap to track k largest = kth largest is min of heap",
      "heapq in Python is a min-heap",
    ],
    timeComplexity: "O(log k)",
    spaceComplexity: "O(k)",
  },
  {
    id: 41,
    slug: "last-stone-weight",
    name: "Last Stone Weight",
    difficulty: "Easy",
    category: "Heap",
    categorySlug: "heap",
    leetcodeUrl: "https://leetcode.com/problems/last-stone-weight/",
    neetcodeUrl: "https://neetcode.io/problems/last-stone-weight",
    solution: `import heapq

def lastStoneWeight(stones: list[int]) -> int:
    # Negate for max-heap behavior
    heap = [-s for s in stones]
    heapq.heapify(heap)

    while len(heap) > 1:
        first = -heapq.heappop(heap)
        second = -heapq.heappop(heap)

        if first != second:
            heapq.heappush(heap, -(first - second))

    return -heap[0] if heap else 0`,
    explanation: [
      "Simulate stone smashing: always pick two heaviest",
      "Use max-heap (negate values for Python's min-heap)",
      "Pop two largest, push difference if not equal",
      "Continue until ≤ 1 stone remains",
    ],
    keyPoints: [
      "O(n log n) time, O(n) space",
      "Python heapq is min-heap; negate for max-heap",
      "Greedy: always smash two heaviest",
      "Return 0 if no stones left",
    ],
    timeComplexity: "O(n log n)",
    spaceComplexity: "O(n)",
  },
  {
    id: 42,
    slug: "kth-largest-element-in-an-array",
    name: "Kth Largest Element in an Array",
    difficulty: "Medium",
    category: "Heap",
    categorySlug: "heap",
    leetcodeUrl: "https://leetcode.com/problems/kth-largest-element-in-an-array/",
    neetcodeUrl: "https://neetcode.io/problems/kth-largest-element-in-an-array",
    solution: `import heapq

def findKthLargest(nums: list[int], k: int) -> int:
    # Min-heap of size k
    heap = []

    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)

    return heap[0]`,
    explanation: [
      "Maintain min-heap of size k",
      "After processing all numbers, heap contains k largest",
      "Root of min-heap is the kth largest",
      "Alternative: QuickSelect for O(n) average",
    ],
    keyPoints: [
      "O(n log k) time, O(k) space",
      "Min-heap keeps k largest; smallest of those is answer",
      "Better than sorting (O(n log n)) when k << n",
      "QuickSelect: O(n) average, O(n²) worst case",
    ],
    timeComplexity: "O(n log k)",
    spaceComplexity: "O(k)",
  },

  // ===== GRAPHS =====
  {
    id: 43,
    slug: "number-of-islands",
    name: "Number of Islands",
    difficulty: "Medium",
    category: "Graphs",
    categorySlug: "graphs",
    leetcodeUrl: "https://leetcode.com/problems/number-of-islands/",
    neetcodeUrl: "https://neetcode.io/problems/count-number-of-islands",
    solution: `def numIslands(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return

        grid[r][c] = '0'  # Mark visited

        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)

    return count`,
    explanation: [
      "Each connected component of '1's is an island",
      "For each unvisited '1', start DFS/BFS",
      "Mark all connected '1's as visited (set to '0')",
      "Count number of DFS/BFS calls = number of islands",
    ],
    keyPoints: [
      "O(m × n) time and space",
      "Mark visited by modifying grid (or use separate set)",
      "4-directional connectivity (up, down, left, right)",
      "Classic flood fill / connected components problem",
    ],
    timeComplexity: "O(m × n)",
    spaceComplexity: "O(m × n)",
  },
  {
    id: 44,
    slug: "max-area-of-island",
    name: "Max Area of Island",
    difficulty: "Medium",
    category: "Graphs",
    categorySlug: "graphs",
    leetcodeUrl: "https://leetcode.com/problems/max-area-of-island/",
    neetcodeUrl: "https://neetcode.io/problems/max-area-of-island",
    solution: `def maxAreaOfIsland(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    max_area = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != 1:
            return 0

        grid[r][c] = 0  # Mark visited

        return 1 + dfs(r+1, c) + dfs(r-1, c) + dfs(r, c+1) + dfs(r, c-1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                max_area = max(max_area, dfs(r, c))

    return max_area`,
    explanation: [
      "Similar to Number of Islands, but track area",
      "DFS returns count of cells in connected component",
      "Sum up 1 (current cell) + area of 4 neighbors",
      "Track maximum area across all islands",
    ],
    keyPoints: [
      "O(m × n) time and space",
      "DFS returns area instead of just marking",
      "Recursive sum: 1 + all connected cells",
      "Same flood fill pattern as Number of Islands",
    ],
    timeComplexity: "O(m × n)",
    spaceComplexity: "O(m × n)",
  },
  {
    id: 45,
    slug: "clone-graph",
    name: "Clone Graph",
    difficulty: "Medium",
    category: "Graphs",
    categorySlug: "graphs",
    leetcodeUrl: "https://leetcode.com/problems/clone-graph/",
    neetcodeUrl: "https://neetcode.io/problems/clone-graph",
    solution: `def cloneGraph(node: 'Node') -> 'Node':
    if not node:
        return None

    cloned = {}

    def dfs(node):
        if node in cloned:
            return cloned[node]

        copy = Node(node.val)
        cloned[node] = copy

        for neighbor in node.neighbors:
            copy.neighbors.append(dfs(neighbor))

        return copy

    return dfs(node)`,
    explanation: [
      "Use hashmap to track original → clone mapping",
      "DFS: create clone, store in map, then clone neighbors",
      "If node already cloned, return existing clone",
      "This handles cycles in the graph",
    ],
    keyPoints: [
      "O(V + E) time and space",
      "Hashmap prevents infinite loops in cyclic graphs",
      "Must clone nodes before their neighbors (or handle later)",
      "Can also use BFS with queue",
    ],
    timeComplexity: "O(V + E)",
    spaceComplexity: "O(V)",
  },
  {
    id: 46,
    slug: "pacific-atlantic-water-flow",
    name: "Pacific Atlantic Water Flow",
    difficulty: "Medium",
    category: "Graphs",
    categorySlug: "graphs",
    leetcodeUrl: "https://leetcode.com/problems/pacific-atlantic-water-flow/",
    neetcodeUrl: "https://neetcode.io/problems/pacific-atlantic-water-flow",
    solution: `def pacificAtlantic(heights: list[list[int]]) -> list[list[int]]:
    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()

    def dfs(r, c, visited, prev_height):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            (r, c) in visited or heights[r][c] < prev_height):
            return

        visited.add((r, c))
        dfs(r + 1, c, visited, heights[r][c])
        dfs(r - 1, c, visited, heights[r][c])
        dfs(r, c + 1, visited, heights[r][c])
        dfs(r, c - 1, visited, heights[r][c])

    for c in range(cols):
        dfs(0, c, pacific, 0)
        dfs(rows - 1, c, atlantic, 0)

    for r in range(rows):
        dfs(r, 0, pacific, 0)
        dfs(r, cols - 1, atlantic, 0)

    return list(pacific & atlantic)`,
    explanation: [
      "Reverse thinking: start from oceans, go uphill",
      "Find all cells that can reach Pacific (top/left edges)",
      "Find all cells that can reach Atlantic (bottom/right edges)",
      "Answer is intersection of both sets",
    ],
    keyPoints: [
      "O(m × n) time and space",
      "Reverse flow: water flows uphill from ocean",
      "Start DFS from ocean borders",
      "Intersection of reachable sets is the answer",
    ],
    timeComplexity: "O(m × n)",
    spaceComplexity: "O(m × n)",
  },
  {
    id: 47,
    slug: "surrounded-regions",
    name: "Surrounded Regions",
    difficulty: "Medium",
    category: "Graphs",
    categorySlug: "graphs",
    leetcodeUrl: "https://leetcode.com/problems/surrounded-regions/",
    neetcodeUrl: "https://neetcode.io/problems/surrounded-regions",
    solution: `def solve(board: list[list[str]]) -> None:
    rows, cols = len(board), len(board[0])

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != 'O':
            return

        board[r][c] = 'T'  # Temporarily mark
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    # Mark border-connected O's
    for r in range(rows):
        dfs(r, 0)
        dfs(r, cols - 1)
    for c in range(cols):
        dfs(0, c)
        dfs(rows - 1, c)

    # Convert: O->X (surrounded), T->O (not surrounded)
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'O':
                board[r][c] = 'X'
            elif board[r][c] == 'T':
                board[r][c] = 'O'`,
    explanation: [
      "O's connected to border cannot be surrounded",
      "Mark all border-connected O's as temporary 'T'",
      "Remaining O's are surrounded - convert to X",
      "Convert T's back to O's",
    ],
    keyPoints: [
      "O(m × n) time and space",
      "Key insight: capture = NOT connected to border",
      "DFS from borders to find safe O's",
      "In-place modification with temporary marker",
    ],
    timeComplexity: "O(m × n)",
    spaceComplexity: "O(m × n)",
  },
  {
    id: 48,
    slug: "course-schedule",
    name: "Course Schedule",
    difficulty: "Medium",
    category: "Graphs",
    categorySlug: "graphs",
    leetcodeUrl: "https://leetcode.com/problems/course-schedule/",
    neetcodeUrl: "https://neetcode.io/problems/course-schedule",
    solution: `def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    graph = {i: [] for i in range(numCourses)}
    for course, prereq in prerequisites:
        graph[course].append(prereq)

    # 0=unvisited, 1=visiting, 2=visited
    state = [0] * numCourses

    def hasCycle(course):
        if state[course] == 1:  # Cycle detected
            return True
        if state[course] == 2:  # Already processed
            return False

        state[course] = 1
        for prereq in graph[course]:
            if hasCycle(prereq):
                return True
        state[course] = 2

        return False

    for course in range(numCourses):
        if hasCycle(course):
            return False

    return True`,
    explanation: [
      "Model as directed graph: course → prerequisites",
      "Can finish all courses if no cycle exists",
      "Use DFS with 3 states: unvisited, visiting, visited",
      "Cycle detected if we revisit a 'visiting' node",
    ],
    keyPoints: [
      "O(V + E) time and space",
      "Cycle detection in directed graph",
      "Three-state approach: unvisited/visiting/visited",
      "Alternative: topological sort with Kahn's algorithm",
    ],
    timeComplexity: "O(V + E)",
    spaceComplexity: "O(V + E)",
  },
  {
    id: 49,
    slug: "course-schedule-ii",
    name: "Course Schedule II",
    difficulty: "Medium",
    category: "Graphs",
    categorySlug: "graphs",
    leetcodeUrl: "https://leetcode.com/problems/course-schedule-ii/",
    neetcodeUrl: "https://neetcode.io/problems/course-schedule-ii",
    solution: `def findOrder(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    graph = {i: [] for i in range(numCourses)}
    for course, prereq in prerequisites:
        graph[course].append(prereq)

    result = []
    state = [0] * numCourses  # 0=unvisited, 1=visiting, 2=visited

    def dfs(course):
        if state[course] == 1:
            return False  # Cycle
        if state[course] == 2:
            return True

        state[course] = 1
        for prereq in graph[course]:
            if not dfs(prereq):
                return False
        state[course] = 2
        result.append(course)

        return True

    for course in range(numCourses):
        if not dfs(course):
            return []

    return result`,
    explanation: [
      "Topological sort: order where prerequisites come first",
      "DFS-based: add course to result after all prereqs processed",
      "If cycle detected, no valid ordering exists",
      "Result is built in correct order (prereqs before course)",
    ],
    keyPoints: [
      "O(V + E) time and space",
      "Topological sort with cycle detection",
      "Add to result after visiting all neighbors",
      "Alternative: Kahn's algorithm (BFS with indegree)",
    ],
    timeComplexity: "O(V + E)",
    spaceComplexity: "O(V + E)",
  },
];

export function getProblemBySlug(slug: string): Problem | undefined {
  return problems.find((p) => p.slug === slug);
}

export function getProblemsByCategory(categorySlug: string): Problem[] {
  return problems.filter((p) => p.categorySlug === categorySlug);
}

export const categories = [
  { name: "Arrays & Hashing", slug: "arrays-hashing", count: 8 },
  { name: "Two Pointers", slug: "two-pointers", count: 4 },
  { name: "Sliding Window", slug: "sliding-window", count: 3 },
  { name: "Stack", slug: "stack", count: 3 },
  { name: "Binary Search", slug: "binary-search", count: 3 },
  { name: "Linked List", slug: "linked-list", count: 6 },
  { name: "Trees", slug: "trees", count: 12 },
  { name: "Heap", slug: "heap", count: 3 },
  { name: "Graphs", slug: "graphs", count: 7 },
];
