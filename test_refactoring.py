#!/usr/bin/env python3
"""
Refactoring Verification Script

Tests:
1. Import from new modular structure
2. Import from backward compatibility wrapper
3. Basic instantiation test
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_new_import():
    """Test importing from new modular structure"""
    print("[TEST 1] Testing new modular import...")
    try:
        from agents.graph import (
            TeamHGraph,
            TeamHState,
            AgentRouting,
            create_team_h_graph,
            get_team_h_graph,
        )
        print("  ✅ Successfully imported from agents.graph")
        return True
    except Exception as e:
        print(f"  ❌ Failed to import from agents.graph: {e}")
        return False


def test_backward_compatibility():
    """Test importing from backward compatibility wrapper"""
    print("\n[TEST 2] Testing backward compatibility import...")
    try:
        from agents.team_h_graph import (
            TeamHGraph,
            create_team_h_graph,
            get_team_h_graph,
        )
        print("  ✅ Successfully imported from agents.team_h_graph")
        return True
    except Exception as e:
        print(f"  ❌ Failed to import from agents.team_h_graph: {e}")
        return False


def test_instantiation():
    """Test basic instantiation (minimal config)"""
    print("\n[TEST 3] Testing TeamHGraph instantiation...")
    try:
        from agents.graph import TeamHGraph

        # Minimal configuration - only enable Manager M
        graph = TeamHGraph(
            enable_manager_i=False,
            enable_manager_m=True,
            enable_manager_s=False,
            enable_manager_t=False,
            model_name="gpt-4o-mini",
            temperature=0.7,
            use_postgres_checkpoint=False,  # Use in-memory for testing
        )
        print("  ✅ Successfully instantiated TeamHGraph")

        # Test visualization
        viz = graph.get_graph_visualization()
        if viz and not viz.startswith("Visualization not available"):
            print("  ✅ Graph visualization works")
        else:
            print(f"  ⚠️ Graph visualization returned: {viz[:50]}...")

        return True
    except Exception as e:
        print(f"  ❌ Failed to instantiate TeamHGraph: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Team-H Graph Refactoring Verification")
    print("=" * 60)

    results = []
    results.append(test_new_import())
    results.append(test_backward_compatibility())
    results.append(test_instantiation())

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("✅ All tests passed! Refactoring successful.")
        return 0
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
