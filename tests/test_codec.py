import unittest
import mlx.core as mx
import numpy as np

from drama3_mlx.codec import DAC, build_ae


class TestCodec(unittest.TestCase):
    def test_dac_instantiation(self):
        dac = build_ae()
        self.assertEqual(dac.sample_rate, 44100)
        self.assertEqual(dac.frame_length, 2048)

    def test_encode_decode_roundtrip(self):
        dac = build_ae()
        # 1 frame = 2048 samples; test 4096 samples (2 frames)
        audio = mx.random.normal((1, 1, 4096))
        codes, lengths = dac.encode(audio)
        self.assertEqual(codes.ndim, 3)
        self.assertEqual(codes.shape[1], 10)  # 1 semantic + 9 acoustic = 10 codebooks
        self.assertEqual(codes.shape[2], 2)   # 2 frames

        decoded, audio_lens = dac.decode(codes, lengths)
        self.assertEqual(decoded.shape[0], 1)
        self.assertEqual(decoded.shape[1], 1)
        self.assertEqual(decoded.shape[2], 4096)


if __name__ == "__main__":
    unittest.main()
