"""
Tests for the dispatch table implementation.

This module tests the complete dispatch table system including:
- Basic dispatch functionality
- Builder pattern usage
- Extension mechanism
- Edge cases and error handling
"""
from sealir.dispatchtable import dispatchtable, DispatchTableBuilder, DispatchTable, Case


def test_dispatchtable_decorator():
    """Test the @dispatchtable decorator functionality."""
    @dispatchtable
    def disp(builder):
        @builder.default
        def default_handler(value):
            # Default case - returns a tuple with "default" and the value
            return "default", value

        @builder.case(lambda x: x == 1)
        def handle_one(value):
            # Case for value == 1 - returns specific response
            return "case 1", value

    # Test basic dispatch behavior
    assert disp(1) == ("case 1", 1)      # Should hit the case for x == 1
    assert disp(None) == ("default", None)  # Should hit the default case
    assert disp(42) == ("default", 42)   # Should hit the default case


def test_dispatch_table_extension():
    """Test the extension mechanism for dispatch tables."""
    @dispatchtable
    def base_disp(builder):
        @builder.default
        def default_handler(value):
            return "default", value

        @builder.case(lambda x: x == 1)
        def handle_one(value):
            return "case 1", value

    @base_disp.extend
    def extended_disp(builder):
        @builder.case(lambda x: x is None)
        def handle_none(value):
            # Case for value == None - returns extended response
            return "extended", value

    # Test extended dispatch behavior
    assert extended_disp(1) == ("case 1", 1)         # Should inherit case for x == 1
    assert extended_disp(None) == ("extended", None)  # Should use new case for x == None
    assert extended_disp(42) == ("default", 42)      # Should use inherited default

    # IMPORTANT: Verify that the parent table is unaffected by extension
    assert base_disp(None) == ("default", None)      # Original should be unchanged


def test_builder_pattern():
    """Test the DispatchTableBuilder class directly."""
    builder = DispatchTableBuilder()

    @builder.case(lambda x: isinstance(x, int), lambda x: x > 0)
    def handle_positive_int(x):
        return f"positive int: {x}"

    @builder.case(lambda x: isinstance(x, str))
    def handle_string(x):
        return f"string: {x}"

    @builder.default
    def handle_other(x):
        return f"other: {x}"

    # Build the dispatch table
    table = builder.build()

    # Test dispatch behavior
    assert table(5) == "positive int: 5"      # matches both int and > 0 conditions
    assert table(-3) == "other: -3"           # matches int but not > 0, so default
    assert table("hello") == "string: hello"  # matches string condition
    assert table([1, 2]) == "other: [1, 2]"  # matches no conditions, so default


def test_multiple_conditions():
    """Test cases with multiple conditions (all must be True)."""
    @dispatchtable
    def multi_cond_disp(builder):
        @builder.case(lambda x: x > 0, lambda x: x % 2 == 0)
        def handle_positive_even(x):
            return f"positive even: {x}"

        @builder.case(lambda x: x > 0, lambda x: x % 2 == 1)
        def handle_positive_odd(x):
            return f"positive odd: {x}"

        @builder.default
        def handle_other(x):
            return f"other: {x}"

    assert multi_cond_disp(4) == "positive even: 4"    # > 0 and even
    assert multi_cond_disp(3) == "positive odd: 3"     # > 0 and odd
    assert multi_cond_disp(-2) == "other: -2"          # even but not > 0
    assert multi_cond_disp(0) == "other: 0"            # even but not > 0


def test_dispatch_table_get_classmethod():
    """Test DispatchTable.get class method."""
    def handler(x):
        return f"handled: {x}"

    def default_handler(x):
        return f"default: {x}"

    cases = [Case(fn=handler, conditions=(lambda x: x == "test",))]
    table = DispatchTable.get(cases, default_handler)

    assert table("test") == "handled: test"
    assert table("other") == "default: other"


def test_builder_get_classmethod():
    """Test DispatchTableBuilder.get class method for copying existing tables."""
    # Create original table
    @dispatchtable
    def original(builder):
        @builder.case(lambda x: x == 1)
        def handle_one(x):
            return "original one"

        @builder.default
        def handle_default(x):
            return "original default"

    # Create builder from existing table
    builder = DispatchTableBuilder.get(original)

    # Add new case
    @builder.case(lambda x: x == 2)
    def handle_two(x):
        return "new two"

    # Build new table
    new_table = builder.build()

    # Test that new table has both original and new functionality
    assert new_table(1) == "original one"     # from original
    assert new_table(2) == "new two"          # newly added
    assert new_table(3) == "original default" # from original

    # Test that original table is unchanged
    assert original(2) == "original default"



def test_case_order_matters():
    """Test that cases are evaluated in order and first match wins."""
    @dispatchtable
    def ordered_disp(builder):
        @builder.case(lambda x: x > 0)
        def handle_positive(x):
            return "positive"

        @builder.case(lambda x: x > 10)
        def handle_big(x):
            return "big"

        @builder.default
        def handle_default(x):
            return "default"

    # First condition (x > 0) should match before second (x > 10) for x = 15
    assert ordered_disp(15) == "positive"
    assert ordered_disp(-5) == "default"


def test_kwargs_support():
    """Test that dispatch works with keyword arguments."""
    @dispatchtable
    def kwargs_disp(builder):
        @builder.case(lambda *args, **kwargs: kwargs.get('special', False))
        def handle_special(*args, **kwargs):
            return f"special: {args}, {kwargs}"

        @builder.default
        def handle_default(*args, **kwargs):
            return f"default: {args}, {kwargs}"

    assert kwargs_disp(1, 2, special=True) == "special: (1, 2), {'special': True}"
    assert kwargs_disp(1, 2, special=False) == "default: (1, 2), {'special': False}"
    assert kwargs_disp(1, 2) == "default: (1, 2), {}"
