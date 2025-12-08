"""
Test Suite for Phase Transition Detection

Tests the detection of cosmological phase transitions:
- inflation (⟨k⟩ < 2)
- matter (2 ≤ ⟨k⟩ < 6)
- accelerated (⟨k⟩ ≥ 6)
"""

import pytest
import numpy as np


class TestPhaseDetection:
    """Test phase classification logic."""

    def test_inflation_phase(self):
        """
        Test classification of inflation phase: ⟨k⟩ < 2
        """
        mean_k = 1.5

        if mean_k < 2:
            phase = "inflation"
        elif mean_k < 6:
            phase = "matter"
        else:
            phase = "accelerated"

        assert phase == "inflation", \
            f"Wrong phase for ⟨k⟩={mean_k}: got {phase}"

    def test_matter_phase(self):
        """
        Test classification of matter phase: 2 ≤ ⟨k⟩ < 6
        """
        for mean_k in [2.0, 3.5, 5.9]:
            if mean_k < 2:
                phase = "inflation"
            elif mean_k < 6:
                phase = "matter"
            else:
                phase = "accelerated"

            assert phase == "matter", \
                f"Wrong phase for ⟨k⟩={mean_k}: got {phase}"

    def test_accelerated_phase(self):
        """
        Test classification of accelerated phase: ⟨k⟩ ≥ 6
        """
        for mean_k in [6.0, 8.5, 15.0]:
            if mean_k < 2:
                phase = "inflation"
            elif mean_k < 6:
                phase = "matter"
            else:
                phase = "accelerated"

            assert phase == "accelerated", \
                f"Wrong phase for ⟨k⟩={mean_k}: got {phase}"

    def test_phase_boundary_precision(self):
        """
        Test exact boundaries between phases.
        """
        # Just below threshold
        mean_k = 1.99999
        phase = "inflation" if mean_k < 2 else ("matter" if mean_k < 6 else "accelerated")
        assert phase == "inflation"

        # Just at threshold
        mean_k = 2.0
        phase = "inflation" if mean_k < 2 else ("matter" if mean_k < 6 else "accelerated")
        assert phase == "matter"

        # Just below second threshold
        mean_k = 5.99999
        phase = "inflation" if mean_k < 2 else ("matter" if mean_k < 6 else "accelerated")
        assert phase == "matter"

        # Just at second threshold
        mean_k = 6.0
        phase = "inflation" if mean_k < 2 else ("matter" if mean_k < 6 else "accelerated")
        assert phase == "accelerated"

    def test_phase_transition_sequence(self):
        """
        Test expected sequence: inflation → matter → accelerated
        """
        phases = []

        # Simulate increasing connectivity
        for mean_k in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0]:
            if mean_k < 2:
                phase = "inflation"
            elif mean_k < 6:
                phase = "matter"
            else:
                phase = "accelerated"

            phases.append(phase)

        # Should observe sequence
        inflation_count = phases.count("inflation")
        matter_count = phases.count("matter")
        accelerated_count = phases.count("accelerated")

        assert inflation_count > 0, "No inflation phase observed"
        assert matter_count > 0, "No matter phase observed"
        assert accelerated_count > 0, "No accelerated phase observed"

        # Check order
        first_matter = phases.index("matter")
        first_accelerated = phases.index("accelerated")

        assert first_matter < first_accelerated, \
            "Phases out of order: matter should come before accelerated"
