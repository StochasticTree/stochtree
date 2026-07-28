"""Unit tests for the JSON key primitives (rename/erase) used by schema migrations."""

from stochtree.serialization import JSONSerializer


def test_rename_and_erase_top_level():
    s = JSONSerializer()
    s.add_scalar("old", 3.5)

    s.rename_field("old", "new")
    assert not s.contains("old")
    assert s.contains("new")
    assert s.get_scalar("new") == 3.5

    s.erase_field("new")
    assert not s.contains("new")


def test_rename_and_erase_subfolder():
    s = JSONSerializer()
    s.add_scalar("old", 7.0, subfolder_name="sub")

    s.rename_field("old", "new", subfolder_name="sub")
    assert not s.contains("old", subfolder_name="sub")
    assert s.contains("new", subfolder_name="sub")
    assert s.get_scalar("new", subfolder_name="sub") == 7.0

    s.erase_field("new", subfolder_name="sub")
    assert not s.contains("new", subfolder_name="sub")


def test_rename_and_erase_missing_is_noop():
    s = JSONSerializer()
    # Neither should raise on an absent field.
    s.rename_field("nope", "whatever")
    s.erase_field("nope")
    assert not s.contains("whatever")
    assert not s.contains("nope")
