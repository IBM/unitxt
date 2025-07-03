import copy
import threading
import time

from unitxt.utils import (
    LRUCache,
    deep_copy,
    is_module_available,
    is_package_installed,
    recursive_copy,
    recursive_deep_copy,
    recursive_shallow_copy,
    remove_numerics_and_quoted_texts,
    shallow_copy,
)

from tests.utils import UnitxtTestCase


class CustomObject:
    def __init__(self, value):
        self.value = value

    def __copy__(self):
        return CustomObject(self.value)

    def __deepcopy__(self, memo):
        return CustomObject(copy.deepcopy(self.value, memo))

    def __eq__(self, other):
        return isinstance(other, CustomObject) and self.value == other.value

    def __repr__(self):
        return f"CustomObject({self.value})"


class TestUtils(UnitxtTestCase):
    def setUp(self):
        # Immutable objects
        self.int_obj = 42
        self.float_obj = 3.14
        self.str_obj = "hello"
        self.tuple_obj = (1, 2, 3)

        # Mutable objects
        self.list_obj = [1, 2, 3]
        self.dict_obj = {"a": 1, "b": 2}
        self.set_obj = {1, 2, 3}

        # Nested structures
        self.nested_list = [1, [2, [3, [4]]]]
        self.nested_dict = {"a": {"b": {"c": {"d": 1}}}}

        # Cyclic structures
        self.cyclic_list = [1, 2, 3]
        self.cyclic_list.append(self.cyclic_list)

        # Custom objects
        self.custom_obj = CustomObject(10)
        self.custom_obj_with_list = CustomObject(self.list_obj)

        # Empty collections
        self.empty_list = []
        self.empty_dict = {}
        self.empty_tuple = ()
        self.empty_set = set()

    def test_recursive_shallow_copy(self):
        # Test nested structures
        nested_list_copy = recursive_shallow_copy(self.nested_list)
        self.assertIsNot(nested_list_copy, self.nested_list)
        self.assertIsNot(nested_list_copy[1], self.nested_list[1])
        self.assertIsNot(nested_list_copy[1][1], self.nested_list[1][1])
        self.assertIsNot(nested_list_copy[1][1][1], self.nested_list[1][1][1])
        self.assertIs(nested_list_copy[0], self.nested_list[0])  # Immutable element
        self.assertIs(
            nested_list_copy[1][0], self.nested_list[1][0]
        )  # Immutable element
        self.assertIs(
            nested_list_copy[1][1][0], self.nested_list[1][1][0]
        )  # Immutable element
        self.assertIs(
            nested_list_copy[1][1][1][0], self.nested_list[1][1][1][0]
        )  # Immutable element

    def test_recursive_deep_copy(self):
        # Use a list with mutable elements
        mutable_list = [[1, 2], [3, 4]]
        mutable_list_copy = recursive_deep_copy(mutable_list)
        self.assertEqual(mutable_list_copy, mutable_list)
        self.assertIsNot(mutable_list_copy, mutable_list)
        self.assertIsNot(
            mutable_list_copy[0], mutable_list[0]
        )  # Inner lists are different instances
        self.assertIsNot(mutable_list_copy[1], mutable_list[1])

        # Test that the inner lists are also deep copied
        for original_item, copied_item in zip(mutable_list, mutable_list_copy):
            self.assertIsNot(copied_item, original_item)

        # Test mutable objects with immutable elements
        list_copy = recursive_deep_copy(self.list_obj)
        self.assertEqual(list_copy, self.list_obj)
        self.assertIsNot(list_copy, self.list_obj)
        # Since elements are immutable, they may be the same instances
        # Do not assert that the elements are different instances

        # Test nested structures
        nested_list_copy = recursive_deep_copy(self.nested_list)
        self.assertEqual(nested_list_copy, self.nested_list)
        self.assertIsNot(nested_list_copy, self.nested_list)
        self.assertIsNot(nested_list_copy[1], self.nested_list[1])

        # Test custom objects
        custom_obj_copy = recursive_deep_copy(self.custom_obj_with_list)
        self.assertEqual(custom_obj_copy, self.custom_obj_with_list)
        self.assertIsNot(custom_obj_copy, self.custom_obj_with_list)
        self.assertIsNot(custom_obj_copy.value, self.custom_obj_with_list.value)

    def test_recursive_copy(self):
        # Test with internal_copy=None
        obj_copy = recursive_copy(self.list_obj)
        self.assertEqual(obj_copy, self.list_obj)  # Contents should be equal
        self.assertIsNot(obj_copy, self.list_obj)  # Should be a new list object

        # Elements should be the same instances since internal_copy is None
        for original_item, copied_item in zip(self.list_obj, obj_copy):
            self.assertIs(copied_item, original_item)

        # Test with a nested structure
        nested_copy = recursive_copy(self.nested_list)
        self.assertEqual(nested_copy, self.nested_list)
        self.assertIsNot(nested_copy, self.nested_list)
        self.assertIs(nested_copy[0], self.nested_list[0])  # Immutable element
        self.assertIsNot(nested_copy[1], self.nested_list[1])  # New list

        # Elements inside nested lists should be the same instances
        self.assertIs(nested_copy[1][0], self.nested_list[1][0])
        self.assertIsNot(nested_copy[1][1], self.nested_list[1][1])

    def test_deep_copy(self):
        # Use a list with mutable elements
        mutable_list = [[1, 2], [3, 4]]
        mutable_list_copy = deep_copy(mutable_list)
        self.assertEqual(mutable_list_copy, mutable_list)
        self.assertIsNot(mutable_list_copy, mutable_list)
        self.assertIsNot(
            mutable_list_copy[0], mutable_list[0]
        )  # Inner lists are different instances
        self.assertIsNot(mutable_list_copy[1], mutable_list[1])

        # Test that the inner lists are also deep copied
        for original_item, copied_item in zip(mutable_list, mutable_list_copy):
            self.assertIsNot(copied_item, original_item)

        # Test mutable objects with immutable elements
        list_copy = deep_copy(self.list_obj)
        self.assertEqual(list_copy, self.list_obj)
        self.assertIsNot(list_copy, self.list_obj)
        # Since elements are immutable, they may be the same instances
        # Do not assert that the elements are different instances

        # Test nested structures
        nested_list_copy = deep_copy(self.nested_list)
        self.assertEqual(nested_list_copy, self.nested_list)
        self.assertIsNot(nested_list_copy, self.nested_list)
        self.assertIsNot(nested_list_copy[1], self.nested_list[1])

        # Test cyclic structures
        cyclic_list_copy = deep_copy(self.cyclic_list)
        self.assertIs(cyclic_list_copy[3], cyclic_list_copy)

        # Test custom objects
        custom_obj_copy = deep_copy(self.custom_obj)
        self.assertEqual(custom_obj_copy, self.custom_obj)
        self.assertIsNot(custom_obj_copy, self.custom_obj)

    def test_shallow_copy(self):
        # Test immutable objects
        int_copy = shallow_copy(self.int_obj)
        self.assertEqual(int_copy, self.int_obj)
        self.assertIs(int_copy, self.int_obj)

        # Test mutable objects
        list_copy = shallow_copy(self.list_obj)
        self.assertEqual(list_copy, self.list_obj)
        self.assertIsNot(list_copy, self.list_obj)
        self.assertIs(list_copy[0], self.list_obj[0])  # Same references

        # Test nested structures
        nested_list_copy = shallow_copy(self.nested_list)
        self.assertEqual(nested_list_copy, self.nested_list)
        self.assertIsNot(nested_list_copy, self.nested_list)
        self.assertIs(nested_list_copy[1], self.nested_list[1])

        # Test cyclic structures
        cyclic_list_copy = shallow_copy(self.cyclic_list)
        self.assertIs(cyclic_list_copy[3], self.cyclic_list)

        # Test custom objects
        custom_obj_copy = shallow_copy(self.custom_obj)
        self.assertEqual(custom_obj_copy, self.custom_obj)
        self.assertIsNot(custom_obj_copy, self.custom_obj)

    def test_edge_cases(self):
        # Empty collections
        empty_list_copy = deep_copy(self.empty_list)
        self.assertIsNot(empty_list_copy, self.empty_list)

        # Objects without copy methods
        class NoCopyClass:
            def __init__(self, value):
                self.value = value

        obj = NoCopyClass(5)
        obj_copy = recursive_copy(obj)
        self.assertIs(obj_copy, obj)

    def test_circular_references(self):
        cyclic_dict = {}
        cyclic_dict["self"] = cyclic_dict

        cyclic_dict_copy = deep_copy(cyclic_dict)
        self.assertIs(cyclic_dict_copy["self"], cyclic_dict_copy)

    def test_named_tuples(self):
        from collections import namedtuple

        Point = namedtuple("Point", "x y")
        p = Point(1, 2)
        p_copy = recursive_deep_copy(p)
        self.assertEqual(p_copy, p)
        self.assertIsNot(p_copy, p)

    def test_custom_class_with_recursive_structure(self):
        class TreeNode:
            def __init__(self, value, left=None, right=None):
                self.value = value
                self.left = left
                self.right = right

            def __eq__(self, other):
                if not isinstance(other, TreeNode):
                    return False
                return (
                    self.value == other.value
                    and self.left == other.left
                    and self.right == other.right
                )

        root = TreeNode(1)
        root.left = root
        root_copy = recursive_deep_copy(root)
        self.assertIs(root_copy.left, root_copy)

    def test_is_package_installed_true(self):
        self.assertTrue(is_package_installed("datasets"))

    def test_is_package_installed_false(self):
        self.assertFalse(is_package_installed("some-non-existent-package-name"))

    def test_is_module_available_true(self):
        self.assertTrue(is_module_available("collections"))

    def test_is_module_available_false(self):
        self.assertFalse(is_module_available("some_non_existent_module"))

    def test_remove_numerics_and_quoted_texts(self):
        test_cases = [
            ("This is a string with numbers 1234", "This is a string with numbers "),
            (
                "This string contains a float 123.45 in it",
                "This string contains a float  in it",
            ),
            (
                "This string contains a 'quoted string' here",
                "This string contains a  here",
            ),
            (
                'This string contains a "double quoted string" here',
                "This string contains a  here",
            ),
            (
                '''This string contains a """triple quoted string""" here''',
                "This string contains a  here",
            ),
            (
                '''Here are some numbers 1234 and floats 123.45, and strings 'single' "double" """triple""" ''',
                "Here are some numbers  and floats , and strings    ",
            ),
            (
                "This string contains no numbers or quoted strings",
                "This string contains no numbers or quoted strings",
            ),
        ]

        for i, (input_str, expected_output) in enumerate(test_cases, 1):
            with self.subTest(i=i):
                result = remove_numerics_and_quoted_texts(input_str)
                self.assertEqual(result, expected_output)

    def test_separate_caches_per_thread(self):
        """Test that each thread maintains its own separate cache."""
        shared_cache = LRUCache(max_size=5)
        results = {}

        def thread_function(thread_id):
            for i in range(3):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                shared_cache[key] = value

            thread_items = []
            for i in range(3):
                key = f"thread_{thread_id}_key_{i}"
                if key in shared_cache:
                    thread_items.append((key, shared_cache[key]))

            results[thread_id] = {"items": thread_items, "cache_len": len(shared_cache)}

        threads = []
        for i in range(3):
            t = threading.Thread(target=thread_function, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        for thread_id in range(3):
            self.assertIn(thread_id, results)
            thread_result = results[thread_id]

            self.assertEqual(thread_result["cache_len"], 3)

            for key, value in thread_result["items"]:
                self.assertTrue(key.startswith(f"thread_{thread_id}_"))
                self.assertTrue(value.startswith(f"thread_{thread_id}_"))

    def test_no_cross_thread_contamination(self):
        """Test that data from one thread doesn't appear in another thread's cache."""
        shared_cache = LRUCache(max_size=10)
        contamination_found = threading.Event()

        def writer_thread():
            for i in range(5):
                shared_cache[f"writer_{i}"] = f"writer_value_{i}"
                time.sleep(0.01)  # Small delay to increase overlap chance

        def reader_thread():
            time.sleep(0.005)
            for _ in range(10):
                for i in range(5):
                    if f"writer_{i}" in shared_cache:
                        contamination_found.set()
                        return
                time.sleep(0.01)

        writer = threading.Thread(target=writer_thread)
        reader = threading.Thread(target=reader_thread)

        writer.start()
        reader.start()

        writer.join()
        reader.join()

        self.assertFalse(
            contamination_found.is_set(), "Cross-thread contamination detected!"
        )

    def test_clear_only_affects_current_thread(self):
        """Test that clearing cache in one thread doesn't affect other threads."""
        shared_cache = LRUCache(max_size=5)
        thread_states = {}

        def thread_with_clear(thread_id):
            shared_cache[f"key_{thread_id}"] = f"value_{thread_id}"

            if thread_id == 1:
                time.sleep(0.05)  # Ensure other threads have written
                shared_cache.clear()
                thread_states[thread_id] = len(shared_cache)
            else:
                time.sleep(0.1)  # Wait for thread 1 to clear
                # Check if our data is still there
                thread_states[thread_id] = (
                    len(shared_cache),
                    f"key_{thread_id}" in shared_cache,
                )

        threads = []
        for i in range(3):
            t = threading.Thread(target=thread_with_clear, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(thread_states[1], 0)

        for thread_id in [0, 2]:
            cache_len, has_key = thread_states[thread_id]
            self.assertEqual(cache_len, 1)
            self.assertTrue(has_key)
