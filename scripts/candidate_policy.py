"""One split-independent candidate rule for complete-proxy experiments.

The publication values are observed first-impression proxies, not true publish
or click times. Callers supply the union of positives in the current request
(user, timestamp, normalized history); never aggregate later user clicks.
"""
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
import random

PROFILE = 'complete_proxy_strict_request_consistent_v1'
TIME_FORMAT = '%m/%d/%Y %I:%M:%S %p'
MANIFEST_NAME = 'consistent_candidate_manifest.json'
RECIPE = {
    'window_hours': 24,
    'window_endpoints': 'strict',
    'exclusion': 'request_history_and_all_current_positives',
    'positive_policy': 'always_keep_own_GT_for_eligible_requests',
    'request_eligibility': 'at_least_one_eligible_negative',
    'zero_negative_policy': 'exclude_derived_request_record_exclusions_preserve_IMP_master',
    'sampling': 'uniform_without_replacement_min_k_N',
    'train_dev_negative_pool_max': 50,
    'test_negative_counts': [30, 90, 150],
    'test_nested_prefixes': True,
    'rng_stream': '{seed}:consistent_v1:{split}:{source_row}:{positive_index}',
    'pool_order': 'publication_proxy_time_then_news_ID',
    'future_user_clicks_used_for_exclusion': False,
}


def parse_timestamp(value):
    """Accept MIND/ISO naive times without silently inventing a timezone."""
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.strptime(value, TIME_FORMAT)
        except ValueError:
            try:
                result = datetime.fromisoformat(value)
            except ValueError as error:
                raise ValueError('Invalid publication/request timestamp') from error
    else:
        raise ValueError('Timestamp must be a naive datetime or supported string')
    if result.tzinfo is not None:
        raise ValueError('Timezone-aware timestamps require an explicit dataset conversion')
    return result


def _ids(values, name):
    if values is None:
        values = ()
    elif isinstance(values, str):
        values = values.split()
    result = tuple(values)
    if any(not isinstance(n, str) or not n or any(c.isspace() for c in n) for n in result):
        raise ValueError('Invalid article ID in ' + name)
    return result


class _EligibleWindow(Sequence):
    """A contiguous time window with sparse exclusions, in the original sampling order."""
    def __init__(self, ids, left, right, excluded_positions):
        self.ids, self.left, self.right = ids, left, right
        self.excluded = sorted(pos for pos in excluded_positions if left <= pos < right)
        # Exclusions at or before a compressed rank shift its index into the source tuple.
        self.skip_ranks = tuple(pos-left-i for i,pos in enumerate(self.excluded))

    def __len__(self):
        return self.right-self.left-len(self.excluded)

    def __getitem__(self, rank):
        if isinstance(rank, slice):
            return tuple(self[i] for i in range(*rank.indices(len(self))))
        if rank < 0:
            rank += len(self)
        if not 0 <= rank < len(self):
            raise IndexError(rank)
        return self.ids[self.left + rank + bisect_right(self.skip_ranks, rank)]

    def __iter__(self):
        start = self.left
        for pos in self.excluded:
            yield from self.ids[start:pos]
            start = pos+1
        yield from self.ids[start:self.right]


class PublicationIndex:
    """Index all known proxies; strict lifetime filtering is the same in every split."""
    def __init__(self, publication):
        if not hasattr(publication, 'items'):
            raise ValueError('Publication proxies must be an ID-to-time mapping')
        values = {}
        for news, stamp in publication.items():
            _ids((news,), 'publication proxy')
            values[news] = parse_timestamp(stamp)
        if not values:
            raise ValueError('Publication proxy is empty')
        self.publication = values
        self.ids = tuple(sorted(values, key=lambda n: (values[n], n)))
        self.times = tuple(values[n] for n in self.ids)
        self.positions = {news: pos for pos, news in enumerate(self.ids)}

    @classmethod
    def from_proxy(cls, path):
        """Load a headerless two-column proxy, rejecting duplicate article IDs."""
        values = {}
        with Path(path).open(encoding='utf-8-sig') as stream:
            for row_number, raw in enumerate(stream, 1):
                parts = raw.rstrip('\r\n').split('\t')
                if len(parts) != 2 or not parts[0] or parts[0] in values:
                    raise ValueError('Invalid or duplicate publication proxy at row ' + str(row_number))
                values[parts[0]] = parse_timestamp(parts[1])
        return cls(values)

    def window_bounds(self, current):
        """Return the half-open index slice for the strict 24-hour availability window."""
        current = parse_timestamp(current)
        return (bisect_right(self.times, current - timedelta(hours=24)),
                bisect_left(self.times, current))

    def eligible_window(self, current, history, all_current_positives):
        """Provide a random-access pool without materializing all eligible articles per request."""
        positives = set(_ids(all_current_positives, 'current request positives'))
        if not positives:
            raise ValueError('A request must contain at least one positive')
        excluded = set(_ids(history, 'request history')) | positives
        left, right = self.window_bounds(current)
        return _EligibleWindow(self.ids,left,right,(self.positions[n] for n in excluded if n in self.positions))

    def eligible(self, current, history, all_current_positives, known_news_ids=None):
        """Return t-24h < proxy < t, excluding history and every current-request GT.

        An empty tuple is an explicit ineligible derived request. The caller must
        retain the IMP master and record its excluded request/positive counts.
        The positive's own lifetime is never tested here.
        """
        pool = tuple(self.eligible_window(current, history, all_current_positives))
        if known_news_ids is not None:
            known = set(_ids(known_news_ids, 'news metadata'))
            if any(n not in known for n in pool):
                raise ValueError('An eligible proxy article has no metadata; do not silently shrink the universe')
        return pool


def sample_negative_prefixes(pool, counts=(30, 90, 150), *, seed=42, split='test', source_row=1, positive_index=0):
    """Draw one permutation prefix; requested k uses min(k,N) unique negatives."""
    if not isinstance(pool, _EligibleWindow):
        pool = _ids(pool, 'eligible negative pool')
        if len(set(pool)) != len(pool):
            raise ValueError('Eligible negative pool contains duplicate article IDs')
    counts = tuple(counts)
    if not counts or any(type(k) is not int or k < 1 for k in counts) or len(set(counts)) != len(counts):
        raise ValueError('Requested negative counts must be distinct positive integers')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer')
    if split not in ('train', 'dev', 'test'):
        raise ValueError('Unknown split')
    if type(source_row) is not int or source_row < 1 or type(positive_index) is not int or positive_index < 0:
        raise ValueError('Source row is one-based; positive index is zero-based')
    rng = random.Random(f'{seed}:consistent_v1:{split}:{source_row}:{positive_index}')
    chosen = tuple(rng.sample(pool, min(max(counts), len(pool))))
    return {k: chosen[:min(k, len(chosen))] for k in counts}


def training_negative_pool(pool, *, seed=42, split='train', source_row=1, positive_index=0):
    """Use the same sampler for train/dev; optimizer negative slots are separate."""
    return sample_negative_prefixes(pool, counts=(50,), seed=seed, split=split,
                                    source_row=source_row, positive_index=positive_index)[50]


def candidate_pairs(positive, negative_ids):
    """Always retain the GT for an eligible request, regardless of its lifetime."""
    positive = _ids((positive,), 'positive')[0]
    negatives = _ids(negative_ids, 'negative candidates')
    if not negatives:
        raise ValueError('Ineligible request: preserve master and record exclusion instead of writing positive-only data')
    if positive in negatives or len(set(negatives)) != len(negatives):
        raise ValueError('Negative candidates contain GT or duplicate article IDs')
    return ((positive, 1),) + tuple((n, 0) for n in negatives)
