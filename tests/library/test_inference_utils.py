import asyncio
import threading

from unitxt.error_utils import UnitxtError
from unitxt.inference import run_coroutine_synchronously

from tests.utils import UnitxtTestCase


async def add_one(x):
    await asyncio.sleep(0)
    return x + 1


class TestRunCoroutineSynchronously(UnitxtTestCase):
    def test_returns_coroutine_result(self):
        self.assertEqual(run_coroutine_synchronously(add_one(1)), 2)

    def test_works_after_asyncio_run(self):
        # asyncio.run() leaves the main thread without a current event loop
        asyncio.run(add_one(0))
        self.assertEqual(run_coroutine_synchronously(add_one(1)), 2)

    def test_works_in_worker_thread(self):
        results = []
        thread = threading.Thread(
            target=lambda: results.append(run_coroutine_synchronously(add_one(1)))
        )
        thread.start()
        thread.join()
        self.assertEqual(results, [2])

    def test_reuses_event_loop_across_calls(self):
        # Like the semaphore of LiteLLMInferenceEngine, which is created once and
        # binds to the event loop it is first contended on.
        semaphore = asyncio.Semaphore(1)

        async def guarded(x):
            async with semaphore:
                return await add_one(x)

        async def contend():
            return await asyncio.gather(guarded(1), guarded(2))

        self.assertEqual(run_coroutine_synchronously(contend()), [2, 3])
        self.assertEqual(run_coroutine_synchronously(contend()), [2, 3])

    def test_raises_informative_error_inside_running_loop(self):
        coroutine = add_one(1)

        async def main():
            run_coroutine_synchronously(coroutine)

        with self.assertRaises(UnitxtError) as cm:
            asyncio.run(main())
        self.assertIn("nest_asyncio", str(cm.exception))
        self.assertIn("asyncio.to_thread", str(cm.exception))
        # the coroutine is closed rather than left un-awaited
        self.assertIsNone(coroutine.cr_frame)
