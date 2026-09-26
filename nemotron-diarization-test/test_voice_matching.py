"""Local policy tests; no downloads or model inference required."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from voice_matching import (
    MODEL_ID, SAMPLE_RATE, choose_match, exclusive_intervals,
    load_profile, normalized, select_clips,
)


class VoiceMatchingTests(unittest.TestCase):
    def test_unknown_ambiguous_and_unique_matches(self):
        self.assertIsNone(choose_match({0: 0.7}, 0.86)[0])
        self.assertIsNone(choose_match({0: 0.92, 1: 0.90}, 0.86)[0])
        self.assertEqual(choose_match({0: 0.94, 1: 0.65}, 0.86)[0], 0)
        self.assertIsNone(choose_match({}, 0.86)[0])

    def test_invalid_scores_and_threshold(self):
        for threshold in (float("nan"), float("inf"), 2):
            with self.assertRaises(ValueError):
                choose_match({0: 0.9}, threshold)
        with self.assertRaises(ValueError):
            choose_match({0: float("nan")}, 0.86)

    def test_overlap_is_subtracted_including_nested_and_crossing(self):
        segments = [
            {"Speaker": 0, "Start": 1, "End": 15},
            {"Speaker": 1, "Start": 0, "End": 2},
            {"Speaker": 1, "Start": 6, "End": 8},
            {"Speaker": 2, "Start": 7, "End": 10},
            {"Speaker": 1, "Start": 14, "End": 20},
        ]
        self.assertEqual(exclusive_intervals(segments, 0), [(2, 6), (10, 14)])

    def test_no_clips_from_silence_or_short_segments(self):
        self.assertEqual(select_clips(np.zeros(5 * SAMPLE_RATE), [(0, 5)]), [])
        self.assertEqual(select_clips(np.ones(5 * SAMPLE_RATE), [(1, 2.9)]), [])

    def test_clips_are_bounded_and_do_not_cross_interval_gaps(self):
        audio = np.ones(50 * SAMPLE_RATE, dtype=np.float32)
        clips = select_clips(audio, [(0, 25), (30, 40)], max_clips=2)
        self.assertEqual([len(c) for c in clips], [10 * SAMPLE_RATE] * 2)
        clips = select_clips(audio, [(0, 4), (10, 14)])
        self.assertEqual([len(c) for c in clips], [4 * SAMPLE_RATE] * 2)

    def test_profile_validation(self):
        profile = {
            "schema_version": 1, "name": "Test", "model_id": MODEL_ID,
            "sample_rate": SAMPLE_RATE, "embedding": [1.0] * 512,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.voice.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            self.assertAlmostEqual(float(np.linalg.norm(load_profile(path)["embedding"])), 1, places=5)
            for key, value in (("model_id", "wrong"), ("embedding", [0.] * 512),
                               ("embedding", [1.] * 10), ("sample_rate", 8000),
                               ("name", ""), ("model_revision", "not-a-commit")):
                path.write_text(json.dumps({**profile, key: value}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_profile(path)

    def test_invalid_embeddings(self):
        for vector in ([float("nan")], [0.0], [[1.0]]):
            with self.assertRaises(ValueError):
                normalized(vector)


if __name__ == "__main__":
    unittest.main()
