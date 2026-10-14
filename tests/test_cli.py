import subprocess
import sys
import unittest


class TestCLI(unittest.TestCase):
    def test_cli_help(self):
        res = subprocess.run(
            [sys.executable, "-m", "drama3_mlx.cli", "--help"],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "/Users/vanch/drama-3-mlx/src"}
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("drama3-mlx", res.stdout)
        self.assertIn("tts", res.stdout)
        self.assertIn("convert", res.stdout)
        self.assertIn("serve", res.stdout)

    def test_cli_tts_help(self):
        res = subprocess.run(
            [sys.executable, "-m", "drama3_mlx.cli", "tts", "--help"],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "/Users/vanch/drama-3-mlx/src"}
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("--ref-audio", res.stdout)
        self.assertIn("--temperature", res.stdout)


if __name__ == "__main__":
    unittest.main()
