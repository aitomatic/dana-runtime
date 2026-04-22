"""
Unit tests for SearchResource.

Tests the Glob and Grep tools to ensure they work correctly with various
parameter combinations, especially the path + pattern interaction.
"""

import asyncio

import pytest

from dana.core.resource.search_resource import SearchResource


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a workspace with nested directory structure for testing.

    Structure:
        tmp_path/
        ├── ontology/
        │   ├── core.owl
        │   ├── domain.owl
        │   └── sub/
        │       └── nested.owl
        ├── data/
        │   ├── file1.json
        │   └── file2.json
        └── readme.md
    """
    # ontology directory
    ontology_dir = tmp_path / "ontology"
    ontology_dir.mkdir()
    (ontology_dir / "core.owl").write_text("core ontology content")
    (ontology_dir / "domain.owl").write_text("domain ontology content")
    sub_dir = ontology_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "nested.owl").write_text("nested ontology content")

    # data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "file1.json").write_text('{"key": "value1"}')
    (data_dir / "file2.json").write_text('{"key": "value2"}')

    # root file
    (tmp_path / "readme.md").write_text("# Readme")

    return tmp_path


@pytest.fixture
def search_resource(tmp_workspace):
    """Create a SearchResource with the tmp workspace as base_path."""
    return SearchResource(resource_id="test-search", base_path=tmp_workspace)


class TestGlobPathInteraction:
    """Test the interaction between pattern and path parameters in Glob.

    Pattern is always relative to path. Do NOT repeat the directory name
    in the pattern when path is already set.

    Correct:   pattern="**/*.owl",  path="ontology"
    Wrong:     pattern="ontology/**/*.owl", path="ontology"  (path doubling)
    """

    def test_glob_redundant_path_in_pattern_returns_no_results(self, search_resource):
        """Verify that pattern='ontology/**' with path='ontology' finds nothing.

        This is the exact tool call from the bug report:
        {
            "function": "Glob",
            "arguments": {
                "pattern": "ontology/**",
                "path": "ontology"
            }
        }

        The pattern is relative to path, so this looks for
        base_path/ontology/ontology/** — which doesn't exist.
        The fix is on the caller side (use pattern="**/*" instead).
        """
        result = asyncio.run(search_resource.glob(pattern="ontology/**", path="ontology"))
        assert "No files found" in result

    def test_glob_correct_usage_all_files(self, search_resource):
        """Correct usage: pattern='**/*' with path='ontology' finds all files."""
        result = asyncio.run(search_resource.glob(pattern="**/*", path="ontology"))
        assert "No files found" not in result
        assert "core.owl" in result
        assert "domain.owl" in result
        assert "nested.owl" in result

    def test_glob_correct_usage_extension_filter(self, search_resource):
        """Correct usage: pattern='**/*.owl' with path='ontology'."""
        result = asyncio.run(search_resource.glob(pattern="**/*.owl", path="ontology"))
        assert "No files found" not in result
        assert "core.owl" in result
        assert "domain.owl" in result
        assert "nested.owl" in result

    def test_glob_subdirectory_via_pattern(self, search_resource):
        """pattern='sub/*' with path='ontology' finds files in sub/."""
        result = asyncio.run(search_resource.glob(pattern="sub/*", path="ontology"))
        assert "No files found" not in result
        assert "nested.owl" in result

    def test_glob_pattern_with_no_path_uses_base(self, search_resource):
        """pattern='ontology/**/*' with path=None searches from base_path."""
        result = asyncio.run(search_resource.glob(pattern="ontology/**/*", path=None))
        assert "No files found" not in result
        assert "core.owl" in result
        assert "domain.owl" in result

    def test_glob_extension_filter_from_base(self, search_resource):
        """pattern='**/*.json' with path=None finds JSON files everywhere."""
        result = asyncio.run(search_resource.glob(pattern="**/*.json", path=None))
        assert "No files found" not in result
        assert "file1.json" in result
        assert "file2.json" in result

    def test_glob_direct_extension_with_path(self, search_resource):
        """pattern='*.owl' with path='ontology' finds top-level OWL files."""
        result = asyncio.run(search_resource.glob(pattern="*.owl", path="ontology"))
        assert "No files found" not in result
        assert "core.owl" in result
        assert "domain.owl" in result
        # nested.owl is in sub/, not matched by *.owl (non-recursive)
        assert "nested.owl" not in result

    def test_glob_nonexistent_path(self, search_resource):
        """Nonexistent path returns an error."""
        result = asyncio.run(search_resource.glob(pattern="**/*", path="nonexistent"))
        assert "Error" in result

    def test_glob_absolute_path(self, search_resource, tmp_workspace):
        """Absolute path bypasses base_path resolution."""
        abs_path = str(tmp_workspace / "ontology")
        result = asyncio.run(search_resource.glob(pattern="*.owl", path=abs_path))
        assert "No files found" not in result
        assert "core.owl" in result

    def test_glob_returns_files_only(self, search_resource):
        """Glob returns only files, not directories."""
        result = asyncio.run(search_resource.glob(pattern="**/*", path="ontology"))
        # "sub" is a directory, should not appear as a standalone entry
        lines = result.strip().splitlines()
        for line in lines:
            assert not line.endswith("/sub"), "Directories should not appear in results"
            # Each result should be a file with an extension
            assert "." in line.rsplit("/", 1)[-1], f"Expected file path, got: {line}"

    def test_glob_redundant_path_with_extension_returns_no_results(self, search_resource):
        """pattern='ontology/*.owl' with path='ontology' is a path-doubling mistake."""
        result = asyncio.run(search_resource.glob(pattern="ontology/*.owl", path="ontology"))
        assert "No files found" in result


class TestGrepPathInteraction:
    """Test the interaction between pattern and path parameters in Grep (Python native)."""

    def test_grep_in_subdirectory(self, search_resource):
        """Grep searching within a specific subdirectory."""
        result = asyncio.run(
            search_resource.grep(
                pattern="ontology content",
                path="ontology",
                output_mode="files_with_matches",
            )
        )
        assert "No matches found" not in result
        assert "core.owl" in result

    def test_grep_with_glob_filter(self, search_resource):
        """Grep with glob filter for file type."""
        result = asyncio.run(
            search_resource.grep(
                pattern="value",
                path="data",
                output_mode="content",
                glob="*.json",
            )
        )
        assert "No matches found" not in result
        assert "value" in result


class TestGrepSingleFileAutoPromotion:
    """When `path` is a single file, files_with_matches is nearly useless
    (returns only the path the caller already has, which is easily misread
    as an empty result). The main Grep tool auto-promotes to content mode
    with an explanatory header.

    Regression: Forge/Watts session aff85fde-... — agent passed a file path
    with default output_mode and concluded the YAML "had no VAV entries",
    when in fact 328 lines matched.
    """

    def test_single_file_default_mode_auto_promotes_to_content(self, search_resource):
        """Default output_mode on a file path → content mode with note."""
        result = asyncio.run(
            search_resource.grep(
                pattern="core ontology",
                path="ontology/core.owl",
            )
        )
        assert "auto-promoted" in result
        assert "files_with_matches" in result
        assert "core ontology content" in result
        # show_line_numbers defaults to True → expect a "1:" prefix on the hit.
        assert "1:core ontology content" in result

    def test_single_file_explicit_content_no_note(self, search_resource):
        """Explicit output_mode='content' should not trigger the promotion note."""
        result = asyncio.run(
            search_resource.grep(
                pattern="core ontology",
                path="ontology/core.owl",
                output_mode="content",
            )
        )
        assert "auto-promoted" not in result
        assert "core ontology content" in result

    def test_single_file_explicit_count_no_promotion(self, search_resource):
        """Explicit output_mode='count' is meaningful for a single file; do not promote."""
        result = asyncio.run(
            search_resource.grep(
                pattern="core ontology",
                path="ontology/core.owl",
                output_mode="count",
            )
        )
        assert "auto-promoted" not in result

    def test_directory_path_default_mode_not_promoted(self, search_resource):
        """files_with_matches on a directory is useful — must not be promoted."""
        result = asyncio.run(
            search_resource.grep(
                pattern="ontology content",
                path="ontology",
            )
        )
        assert "auto-promoted" not in result
        assert "core.owl" in result

    def test_single_file_no_matches_still_notes_promotion(self, search_resource):
        """Even when no matches, the promotion note should appear so the
        caller understands why the output looks different."""
        result = asyncio.run(
            search_resource.grep(
                pattern="xyznotfoundxyz",
                path="ontology/core.owl",
            )
        )
        assert "auto-promoted" in result
        assert "No matches found" in result

    def test_single_file_absolute_path_auto_promotes(self, search_resource, tmp_workspace):
        """Absolute-path single-file input also triggers auto-promotion."""
        abs_path = str(tmp_workspace / "ontology" / "core.owl")
        result = asyncio.run(
            search_resource.grep(
                pattern="core ontology",
                path=abs_path,
            )
        )
        assert "auto-promoted" in result
        assert "core ontology content" in result
