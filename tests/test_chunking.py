import unittest
import mlx.core as mx
from drama3_mlx.chunking import (
    Drama3History,
    parse_speaker_segments,
    split_by_length,
    split_drama3_text,
    utf8_len,
)

class TestChunking(unittest.TestCase):
    def test_short_text(self):
        res = split_by_length("Hello world", 100)
        self.assertEqual(res, ["Hello world"])

    def test_sentence_split(self):
        text = "Hello world! This is a test. Another sentence."
        chunks = split_by_length(text, 25)
        self.assertGreater(len(chunks), 1)

    def test_multi_speaker_tags(self):
        dialogue = "<|speaker:0|> Hello there! <|speaker:1|> General Kenobi!"
        chunks = split_drama3_text(dialogue, max_bytes=200)
        self.assertTrue(any("<|speaker:0|>" in c for c in chunks))
        self.assertTrue(any("<|speaker:1|>" in c for c in chunks))

    def test_history(self):
        h = Drama3History(recent_turns=2)
        h.push("chunk 1", mx.zeros((10, 5)))
        h.push("chunk 2", mx.zeros((10, 5)))
        h.push("chunk 3", mx.zeros((10, 5)))
        cands = list(h.candidates())
        self.assertGreaterEqual(len(cands), 1)
        # First chunk is pinned
        self.assertEqual(h.pinned[0], "chunk 1")

if __name__ == "__main__":
    unittest.main()
