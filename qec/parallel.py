# qec/parallel.py

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from rich.progress import Progress


# Run independent tasks in separate processes and preserve task order.
def parallel_map(function, tasks, num_workers=None, progress_description=""):
    tasks = list(tasks)

    if not tasks:
        return []

    workers = min(num_workers or os.cpu_count() or 1, len(tasks))
    results = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=workers) as executor, Progress() as progress:
        progress_task = progress.add_task(progress_description, total=len(tasks))
        futures = {
            executor.submit(function, task): index
            for index, task in enumerate(tasks)
        }

        for future in as_completed(futures):
            results[futures[future]] = future.result()
            progress.advance(progress_task)

    return results
