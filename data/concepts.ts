export interface Concept {
  slug: string;
  name: string;
  whenToUse: string[];
  commonPatterns: string[];
  keyInsights: string[];
  timeComplexityNotes: string;
  codeTemplate: string;
}

export const concepts: Record<string, Concept> = {
  "arrays-hashing": {
    slug: "arrays-hashing",
    name: "Arrays & Hashing",
    whenToUse: [
      "Need O(1) average lookup, insert, or delete",
      "Counting frequency of elements",
      "Finding duplicates or pairs that sum to target",
      "Grouping elements by some property",
      "Need to check if element exists quickly",
    ],
    commonPatterns: [
      "Hashmap/set for O(1) lookups - store seen values",
      "Counter/frequency map - count occurrences",
      "Two-pass vs one-pass - sometimes one pass is possible",
      "Tuple as hashable key - for grouping (e.g., sorted string for anagrams)",
      "Prefix/suffix products - when division isn't allowed",
    ],
    keyInsights: [
      "Trade space for time: O(n) extra space often enables O(n) time",
      "Sets for existence checks, maps for value lookups",
      "Sorting can help but costs O(n log n)",
      "Consider what to use as the key in your hashmap",
    ],
    timeComplexityNotes:
      "HashMap operations are O(1) average, O(n) worst case. Hash collisions are rare with good hash functions.",
    codeTemplate: `# Hashmap for O(1) lookup
seen = {}  # or set() for existence only

for i, num in enumerate(arr):
    # Check if complement/target exists
    if target - num in seen:
        return [seen[target - num], i]

    # Store value -> index mapping
    seen[num] = i

# Frequency counter pattern
count = {}
for item in arr:
    count[item] = count.get(item, 0) + 1`,
  },
  "two-pointers": {
    slug: "two-pointers",
    name: "Two Pointers",
    whenToUse: [
      "Array/string is sorted or has some order",
      "Searching for pairs that satisfy a condition",
      "Comparing elements from both ends",
      "Partitioning or rearranging in-place",
      "Finding subarrays that meet criteria",
    ],
    commonPatterns: [
      "Opposite ends - start from both ends, move inward",
      "Same direction - slow/fast pointers for different speeds",
      "Sliding window variant - expand/contract based on condition",
      "Three pointers - fix one, two-pointer on rest (3Sum)",
      "Partition - separate elements by criteria",
    ],
    keyInsights: [
      "Works because sorted order gives directional hints",
      "Moving one pointer eliminates multiple possibilities",
      "Often reduces O(n²) brute force to O(n)",
      "For unsorted arrays, consider sorting first if allowed",
    ],
    timeComplexityNotes:
      "Usually O(n) for single traversal. If sorting needed first, O(n log n). 3Sum is O(n²) due to nested iteration.",
    codeTemplate: `# Two pointers from opposite ends
left, right = 0, len(arr) - 1

while left < right:
    curr_sum = arr[left] + arr[right]

    if curr_sum == target:
        return [left, right]
    elif curr_sum < target:
        left += 1   # Need larger sum
    else:
        right -= 1  # Need smaller sum

# Three pointers (3Sum pattern)
arr.sort()
for i in range(len(arr) - 2):
    left, right = i + 1, len(arr) - 1
    while left < right:
        # ... same logic`,
  },
  "sliding-window": {
    slug: "sliding-window",
    name: "Sliding Window",
    whenToUse: [
      "Finding subarray/substring that satisfies constraint",
      "Min/max length subarray with given property",
      "Contiguous sequence problems",
      "When brute force would check all O(n²) subarrays",
      "Fixed or variable size window needed",
    ],
    commonPatterns: [
      "Fixed size - window of exact size k",
      "Variable size - expand until invalid, then shrink",
      "Frequency map in window - track element counts",
      "Max/min tracking - often with auxiliary data structure",
      "Two conditions - when to expand vs when to shrink",
    ],
    keyInsights: [
      "Window maintains valid state as it slides",
      "Right pointer expands, left pointer contracts",
      "Update answer when window is valid",
      "Key question: what makes window invalid?",
    ],
    timeComplexityNotes:
      "O(n) - each element enters and leaves window at most once. Inner operations should be O(1).",
    codeTemplate: `# Variable size sliding window
left = 0
result = 0

for right in range(len(arr)):
    # Expand: add arr[right] to window state

    # Shrink: while window is invalid
    while window_is_invalid():
        # Remove arr[left] from window state
        left += 1

    # Update result with current valid window
    result = max(result, right - left + 1)

# Fixed size window (size k)
for right in range(len(arr)):
    # Add arr[right] to window
    if right >= k - 1:
        # Window is full, process it
        # Remove arr[right - k + 1] (leftmost)`,
  },
  stack: {
    slug: "stack",
    name: "Stack",
    whenToUse: [
      "Matching pairs (parentheses, tags)",
      "Finding next/previous greater/smaller element",
      "Maintaining decreasing/increasing sequence",
      "Undo operations or backtracking",
      "Expression evaluation",
    ],
    commonPatterns: [
      "Monotonic stack - maintain sorted order for next greater/smaller",
      "Matching pairs - push opening, pop on closing",
      "Track minimums - auxiliary stack for O(1) min",
      "Nested structure parsing - recursion simulation",
      "Postfix/prefix evaluation",
    ],
    keyInsights: [
      "LIFO: most recent item accessed first",
      "Monotonic stack: pop elements that break monotonicity",
      "Each element pushed and popped at most once = O(n)",
      "Stack often replaces recursion",
    ],
    timeComplexityNotes:
      "Push and pop are O(1). Monotonic stack is O(n) total despite nested loops - each element pushed/popped once.",
    codeTemplate: `# Matching pairs (parentheses)
stack = []
pairs = {')': '(', ']': '[', '}': '{'}

for char in s:
    if char in pairs:  # Closing bracket
        if not stack or stack[-1] != pairs[char]:
            return False
        stack.pop()
    else:  # Opening bracket
        stack.append(char)

return len(stack) == 0

# Monotonic stack (next greater element)
stack = []  # Store indices
result = [-1] * len(arr)

for i, num in enumerate(arr):
    while stack and arr[stack[-1]] < num:
        idx = stack.pop()
        result[idx] = num  # num is next greater
    stack.append(i)`,
  },
  "binary-search": {
    slug: "binary-search",
    name: "Binary Search",
    whenToUse: [
      "Sorted array or search space",
      "Finding exact value or boundary",
      "Minimizing/maximizing value that satisfies condition",
      "Search space can be halved each step",
      "Answer has monotonic property (once true, stays true)",
    ],
    commonPatterns: [
      "Classic search - find exact value",
      "Left boundary - first occurrence or insertion point",
      "Right boundary - last occurrence",
      "Condition-based - search on answer space",
      "Rotated array - determine which half is sorted",
    ],
    keyInsights: [
      "Works on any monotonic search space, not just arrays",
      "left + (right - left) // 2 avoids overflow",
      "Be careful with boundary conditions: < vs <=, mid+1 vs mid",
      "Rotated array: one half is always sorted",
    ],
    timeComplexityNotes:
      "O(log n) - halving search space each iteration. Number of iterations is log₂(n).",
    codeTemplate: `# Classic binary search
left, right = 0, len(arr) - 1

while left <= right:
    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        left = mid + 1
    else:
        right = mid - 1

return -1  # Not found

# Search for left boundary / insertion point
while left < right:
    mid = left + (right - left) // 2
    if condition(mid):
        right = mid      # mid could be answer
    else:
        left = mid + 1   # mid is too small`,
  },
  "linked-list": {
    slug: "linked-list",
    name: "Linked List",
    whenToUse: [
      "Frequent insertions/deletions at known positions",
      "Unknown size or dynamic size needed",
      "Need to maintain insertion order with O(1) insert",
      "Implementing other data structures (stack, queue, LRU)",
      "When array resizing is too costly",
    ],
    commonPatterns: [
      "Dummy head - simplifies edge cases for head operations",
      "Fast/slow pointers - find middle, detect cycle",
      "Reversal - iterative with prev/curr/next pointers",
      "Merge - combine sorted lists with dummy head",
      "Two pointers with gap - remove nth from end",
    ],
    keyInsights: [
      "No random access - must traverse to reach position",
      "Dummy head simplifies insert/delete at head",
      "Draw diagrams! Pointer manipulation is error-prone",
      "Slow = 1 step, Fast = 2 steps → slow ends at middle",
    ],
    timeComplexityNotes:
      "Access is O(n). Insert/delete at known position is O(1). Finding position is O(n). Space is O(1) extra for most operations.",
    codeTemplate: `# Reverse a linked list
prev, curr = None, head

while curr:
    next_temp = curr.next  # Save next
    curr.next = prev       # Reverse pointer
    prev = curr            # Move prev forward
    curr = next_temp       # Move curr forward

return prev  # New head

# Fast/slow pointers (find middle, detect cycle)
slow, fast = head, head

while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    # If cycle: slow == fast at some point

# slow is at middle when fast reaches end

# Dummy head pattern
dummy = ListNode(0, head)
curr = dummy
# ... operations
return dummy.next`,
  },
  trees: {
    slug: "trees",
    name: "Trees",
    whenToUse: [
      "Hierarchical data or recursive structure",
      "Need O(log n) search, insert, delete (BST)",
      "Processing nodes in specific order (traversals)",
      "Problems involving paths, ancestors, subtrees",
      "Divide and conquer on tree structure",
    ],
    commonPatterns: [
      "DFS traversals - preorder, inorder, postorder",
      "BFS/level order - queue-based, process by levels",
      "Recursive return value - return computed property up",
      "Pass information down - parameters carry context",
      "BST property - left < root < right for search/validation",
    ],
    keyInsights: [
      "Most tree problems are recursive",
      "Think: what info do I need from subtrees?",
      "What info do I pass down to children?",
      "Inorder of BST gives sorted order",
      "Height O(log n) balanced, O(n) worst case",
    ],
    timeComplexityNotes:
      "Traversal is O(n). BST operations O(h) where h is height. Balanced BST: O(log n). Space is O(h) for recursion stack.",
    codeTemplate: `# DFS recursive pattern
def dfs(node):
    if not node:
        return base_case  # e.g., 0, True, None

    # Process current node (preorder)
    left_result = dfs(node.left)
    right_result = dfs(node.right)
    # Process current node (postorder)

    return combine(left_result, right_result)

# BFS level order traversal
from collections import deque
queue = deque([root])

while queue:
    level_size = len(queue)
    for _ in range(level_size):
        node = queue.popleft()
        # Process node
        if node.left:  queue.append(node.left)
        if node.right: queue.append(node.right)`,
  },
  heap: {
    slug: "heap",
    name: "Heap / Priority Queue",
    whenToUse: [
      "Need quick access to min or max element",
      "K largest/smallest elements",
      "Merging K sorted lists",
      "Scheduling or processing by priority",
      "Streaming data - maintain running statistics",
    ],
    commonPatterns: [
      "Top K elements - min heap of size k for k largest",
      "K-way merge - heap of list heads",
      "Two heaps - max heap + min heap for median",
      "Simulation - process events in order",
      "Greedy with priority - always take best available",
    ],
    keyInsights: [
      "Python heapq is min heap - negate for max heap",
      "Min heap of size k → kth largest is heap[0]",
      "heappush and heappop are O(log n)",
      "heapify is O(n), not O(n log n)",
    ],
    timeComplexityNotes:
      "Insert: O(log n). Extract min/max: O(log n). Peek: O(1). Build heap: O(n). Finding kth largest: O(n log k).",
    codeTemplate: `import heapq

# Min heap (default in Python)
heap = []
heapq.heappush(heap, value)
smallest = heapq.heappop(heap)
peek = heap[0]

# Max heap (negate values)
heapq.heappush(heap, -value)
largest = -heapq.heappop(heap)

# Top K largest (use min heap of size k)
heap = []
for num in nums:
    heapq.heappush(heap, num)
    if len(heap) > k:
        heapq.heappop(heap)  # Remove smallest

# heap[0] is kth largest`,
  },
  graphs: {
    slug: "graphs",
    name: "Graphs",
    whenToUse: [
      "Entities with connections/relationships",
      "Finding connected components",
      "Shortest paths or reachability",
      "Detecting cycles",
      "Ordering with dependencies (topological sort)",
    ],
    commonPatterns: [
      "DFS - explore deep, backtrack (islands, cycle detection)",
      "BFS - explore level by level (shortest path unweighted)",
      "Topological sort - ordering with dependencies",
      "Union-Find - dynamic connectivity, cycle in undirected",
      "Grid as graph - 4 or 8 directional neighbors",
    ],
    keyInsights: [
      "Choose representation: adjacency list vs matrix",
      "Track visited to avoid infinite loops",
      "DFS uses stack (or recursion), BFS uses queue",
      "Grid traversal is graph traversal with implicit edges",
      "Topological sort only for DAGs (directed acyclic)",
    ],
    timeComplexityNotes:
      "DFS and BFS are O(V + E). Building adjacency list is O(E). Grid: V = m×n, E = up to 4×m×n.",
    codeTemplate: `# DFS on grid (islands pattern)
def dfs(r, c):
    if (r < 0 or r >= rows or c < 0 or c >= cols
        or grid[r][c] != '1'):
        return

    grid[r][c] = '0'  # Mark visited
    dfs(r + 1, c)
    dfs(r - 1, c)
    dfs(r, c + 1)
    dfs(r, c - 1)

# BFS shortest path
from collections import deque
queue = deque([(start, 0)])  # (node, distance)
visited = {start}

while queue:
    node, dist = queue.popleft()
    if node == target:
        return dist
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append((neighbor, dist + 1))`,
  },
};

export function getConceptBySlug(slug: string): Concept | undefined {
  return concepts[slug];
}
