import unittest
import time
from concurrent.futures import ThreadPoolExecutor
from actuator.plot_cache import PlotCache


class CacheTests(unittest.TestCase):
    def test_singleflight_lru_and_retry(self):
        c = PlotCache(2)
        calls = []

        def compute():
            calls.append(1)
            time.sleep(0.02)
            return 7

        with ThreadPoolExecutor(4) as pool:
            self.assertEqual(
                list(pool.map(lambda _: c.get("a", compute), range(4))), [7] * 4
            )
        self.assertEqual(len(calls), 1)
        c.get("b", lambda: 2)
        c.get("c", lambda: 3)
        self.assertNotIn("a", c.cache)
        with self.assertRaises(ValueError):
            c.get("error", lambda: (_ for _ in ()).throw(ValueError("bad")))
        self.assertEqual(c.get("error", lambda: 4), 4)
        c.pool.shutdown()
