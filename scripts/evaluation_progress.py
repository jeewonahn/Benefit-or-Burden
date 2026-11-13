"""Adapt request-level test progress to whole files, shards and frozen evaluators."""
from inspect import signature
from pathlib import Path

from experiment_progress import test_progress


def evaluate_test_file(evaluator, behavior_path, prefix, protocol, *, start_row=0, total_rows=None):
    """Report ranked requests; only the auxiliary score counter is local to a shard."""
    def report(ranked, file_rows, *, scored=None, score_total=None):
        test_progress(protocol, start_row + ranked, total_rows if total_rows is not None else file_rows,
                      scored=scored, score_total=score_total)

    # Evaluation-only recovery can be bound to an older, immutable evaluator.
    method = evaluator.evaluate_file
    supports_progress = 'progress' in signature(method).parameters
    if not supports_progress and total_rows is None:
        with Path(behavior_path).open('rb') as source:
            total_rows = sum(1 for line in source)
    if total_rows is not None:
        report(0, total_rows)
    if supports_progress:
        result = method(behavior_path, prefix, progress=report)
    else:
        result = method(behavior_path, prefix)
    report(result['rows'], result['rows'])
    return result
