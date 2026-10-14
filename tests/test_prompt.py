import unittest
import mlx.core as mx
from types import SimpleNamespace
from drama3_mlx.config import Drama3Config
from drama3_mlx.prompt import (
    build_drama3_prompt,
    validate_drama3_text,
    with_speaker_tag,
    InvalidInputError,
)

class MockTokenizer:
    def encode(self, text: str):
        # Split into fake token ids
        return SimpleNamespace(ids=[hash(c) % 1000 for c in text])

class TestPrompt(unittest.TestCase):
    def test_with_speaker_tag(self):
        t1 = "Hello"
        self.assertEqual(with_speaker_tag(t1), "<|speaker:0|> Hello")
        t2 = "<|speaker:3|> Hi"
        self.assertEqual(with_speaker_tag(t2), "<|speaker:3|> Hi")

    def test_validate(self):
        with self.assertRaises(InvalidInputError):
            validate_drama3_text("   ", "test")
        with self.assertRaises(InvalidInputError):
            validate_drama3_text("<|im_start|> user", "test")
        with self.assertRaises(InvalidInputError):
            validate_drama3_text("<|speaker:99|> hi", "test")

    def test_build_prompt_shape(self):
        cfg = Drama3Config()
        tokenizer = MockTokenizer()
        rows = build_drama3_prompt("Hello world", tokenizer, cfg)
        self.assertEqual(rows.ndim, 2)
        self.assertEqual(rows.shape[1], 11)  # 1 + 10 codebooks

if __name__ == "__main__":
    unittest.main()
