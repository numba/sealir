from sealir.dispatchtable import dispatchtable


def test_dispatchtable_subclass_init():
    """Test that dispatch tables are properly copied for subclasses.

    This test verifies that subclasses get their own independent dispatch tables
    rather than sharing the parent's table, which is crucial for allowing
    subclasses to add their own cases without affecting the parent class.
    """

    # Create a base class with a dispatch table
    class Dispatch:
        def __init_subclass__(cls):
            # CRITICAL: Copy the dispatch table for each subclass
            # This ensures subclasses get their own independent dispatch table
            cls.dispatch = cls.dispatch.copy()

        @dispatchtable
        def dispatch(self, value):
            # Default case - returns a tuple with "default" and the value
            return "default", value

        @dispatch.case(lambda _, x: x == 1)
        def _(self, value):
            # Case for value == 1 - returns the class and value
            return self.__class__, value

    # Test the base class dispatch behavior
    disp = Dispatch()
    assert disp.dispatch(disp, 1) == (Dispatch, 1)      # Should hit the case for x == 1
    assert disp.dispatch(disp, None) == ("default", None)  # Should hit the default case

    # Create a subclass that inherits the dispatch table
    class Subclass(Dispatch):
        pass

    # Add a new case to the subclass's dispatch table (not the parent's)
    @Subclass.dispatch.case(lambda _, x: x == None)
    def _(self, value):
        # Case for value == None - returns the class and value
        return self.__class__, value

    # Test the subclass dispatch behavior
    sub = Subclass()
    assert sub.dispatch(sub, 1) == (Subclass, 1)       # Should inherit case for x == 1
    assert sub.dispatch(sub, None) == (Subclass, None) # Should use new case for x == None

    # IMPORTANT: Verify that the parent class is unaffected by subclass changes
    # The parent should still use its default case for None, not the subclass case
    assert disp.dispatch(disp, None) == ("default", None)
