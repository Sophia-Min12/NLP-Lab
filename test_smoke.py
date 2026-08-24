"""Smoke test: keeps CI green from the very first push.

첫 푸시부터 CI 배지를 초록색으로 유지하기 위한 스모크 테스트.
(pytest exits non-zero when it collects zero tests — this file guarantees
there is always at least one.)
"""

import sys
import unittest


class TestEnvironment(unittest.TestCase):
    def test_python_version(self):
        self.assertGreaterEqual(sys.version_info, (3, 10))


if __name__ == "__main__":
    unittest.main()
