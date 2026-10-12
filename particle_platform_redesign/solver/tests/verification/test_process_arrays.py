"""Host transport keeps array ownership while workers reuse prepared facts."""

import copyreg
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import pytest

from chamber_particles.cpu import (
    ProcessArrayReader,
    ProcessArrayStore,
    process_array_views,
)


class _ArrayPair(NamedTuple):
    root: np.ndarray
    negative: np.ndarray


@dataclass
class _ArrayFacts:
    root: np.ndarray
    views: Mapping[str, np.ndarray]
    pair: _ArrayPair
    repeated: list[object]


def test_shared_array_graph_preserves_layout_identity_and_mutation() -> None:
    root = np.arange(48, dtype=np.float64).reshape(6, 8).copy()
    fortran = np.asfortranarray(np.arange(24, dtype=np.float64).reshape(4, 6))
    negative = root[::-1, 1::2]
    readonly = root[:, :2]
    readonly.setflags(write=False)
    empty = root[:0, :2]
    repeated: list[object] = [root, root]
    repeated.append(repeated)
    graph = _ArrayFacts(
        root,
        MappingProxyType(
            {"fortran": fortran, "negative": negative, "readonly": readonly, "empty": empty}
        ),
        _ArrayPair(root, negative),
        repeated,
    )
    expected_views = {id(value) for value in (root, fortran, negative, readonly, empty)}
    assert {id(value) for value in process_array_views(graph)} == expected_views
    ordinary_pickle_dispatch = dict(copyreg.dispatch_table)

    with ProcessArrayStore() as store:
        parent_root = store.share(root)
        store.share(fortran)
        reserved = store.shared_bytes
        payload = store.dumps(graph)
        same_layout_payload = store.describe(graph)
        assert store.shared_bytes == reserved
        assert reserved >= root.nbytes + fortran.nbytes
        assert store.private_bytes == 0
        assert copyreg.dispatch_table == ordinary_pickle_dispatch

        with ProcessArrayReader() as first, ProcessArrayReader() as second:
            loaded = first.loads(payload)
            other = second.loads(same_layout_payload)
            assert isinstance(loaded, _ArrayFacts)
            assert isinstance(other, _ArrayFacts)
            assert isinstance(loaded.views, MappingProxyType)
            assert loaded.root is loaded.pair.root is loaded.repeated[0] is loaded.repeated[1]
            assert loaded.pair.negative is loaded.views["negative"]
            assert loaded.repeated[2] is loaded.repeated
            assert loaded.views["fortran"].flags.f_contiguous
            assert loaded.views["negative"].strides == negative.strides
            assert loaded.views["empty"].shape == empty.shape
            assert loaded.views["empty"].strides == empty.strides
            np.testing.assert_array_equal(loaded.root, root)
            np.testing.assert_array_equal(loaded.views["fortran"], fortran)
            np.testing.assert_array_equal(loaded.views["negative"], negative)
            assert np.shares_memory(loaded.root, loaded.views["negative"])
            assert np.shares_memory(loaded.root, loaded.views["readonly"])
            assert not loaded.views["readonly"].flags.writeable
            with pytest.raises(ValueError):
                loaded.views["readonly"][0, 0] = 1.0

            loaded.views["negative"][0, 0] = -37.125
            assert loaded.root[5, 1] == parent_root[5, 1] == other.root[5, 1] == -37.125
            assert root[5, 1] == 41.0
            again = first.loads(payload)
            assert isinstance(again, _ArrayFacts)
            assert again.root is loaded.root
            assert store.shared_bytes == reserved
            store.release_views()
            first.release_views()
            refreshed = first.loads(store.describe(graph))
            assert isinstance(refreshed, _ArrayFacts)
            assert np.shares_memory(refreshed.root, loaded.root)
            assert refreshed.root is refreshed.pair.root is refreshed.repeated[0]
            assert refreshed.root[5, 1] == -37.125
            assert store.shared_bytes == reserved


def test_private_array_root_is_copied_once_per_reader_with_shared_facts() -> None:
    private = np.arange(24, dtype=np.float64).reshape(4, 6).copy()
    negative = private[:, ::-1]
    shared_source = np.arange(12, dtype=np.int64)

    with ProcessArrayStore() as store:
        parent_shared = store.share(shared_source)
        payload = store.dumps(
            (private, negative, private, {"shared": parent_shared}),
            private_arrays=(negative,),
        )
        reserved = store.shared_bytes
        assert store.private_bytes == private.nbytes
        assert reserved >= private.nbytes + shared_source.nbytes

        with ProcessArrayReader() as first, ProcessArrayReader() as second:
            one = first.loads(payload)
            two = second.loads(payload)
            assert isinstance(one, tuple)
            assert isinstance(two, tuple)
            assert one[0] is one[2]
            assert np.shares_memory(one[0], one[1])
            assert not np.shares_memory(one[0], two[0])
            one[1][0, 0] = -13.0
            assert one[0][0, 5] == -13.0
            assert two[0][0, 5] == private[0, 5] == 5.0
            again = first.loads(payload)
            assert isinstance(again, tuple)
            assert again[0] is one[0]
            assert again[0][0, 5] == -13.0
            one[3]["shared"][1] = 777
            assert two[3]["shared"][1] == parent_shared[1] == 777
            assert shared_source[1] == 1
            assert store.private_bytes == private.nbytes
            assert store.shared_bytes == reserved

            store.release_private_templates()
            assert store.private_bytes == 0
            assert store.shared_bytes < reserved
            again = first.loads(payload)
            assert isinstance(again, tuple)
            assert again[0] is one[0]
            assert again[0][0, 5] == -13.0
            first.release_views()
            refreshed = first.loads(payload)
            assert isinstance(refreshed, tuple)
            assert refreshed[0] is not one[0]
            assert np.shares_memory(refreshed[0], one[0])
            assert np.shares_memory(refreshed[0], refreshed[1])
            assert refreshed[0][0, 5] == -13.0
            two[3]["shared"][2] = 888
            assert parent_shared[2] == one[3]["shared"][2] == 888
            store.release_private_templates()
            for describe in (store.share, store.describe, store.dumps):
                with pytest.raises(RuntimeError):
                    describe(private)
            with pytest.raises(RuntimeError):
                store.release_views()
            with ProcessArrayReader() as newcomer:
                with pytest.raises(FileNotFoundError):
                    newcomer.loads(payload)


def test_invalid_views_and_failed_loads_close_only_their_transport_owner() -> None:
    valid = np.arange(8, dtype=np.float64)
    shrinking_root = np.arange(8, dtype=np.float64)
    stale_view = shrinking_root[4:]
    shrinking_root.resize(2, refcheck=False)
    unsupported = (
        np.asarray([object()], dtype=object),
        np.zeros(2, dtype=[("value", "f8")]),
        np.frombuffer(bytearray(32), dtype=np.float64),
        np.ndarray((2, 2), dtype=np.float64, strides=(8, 8)),
        stale_view,
    )
    for invalid in unsupported:
        store = ProcessArrayStore()
        with pytest.raises(ValueError):
            store.dumps((valid, invalid))
        assert store.shared_bytes == 0
        with pytest.raises(RuntimeError):
            store.share(valid)
        with pytest.raises(RuntimeError):
            store.describe(valid)
        store.close()
        store.close()

    with ProcessArrayStore() as store:
        payload = store.describe(valid)
        with ProcessArrayReader() as reader:
            loaded = reader.loads(payload)
            np.testing.assert_array_equal(loaded, valid)
            del loaded
            with pytest.raises(pickle.UnpicklingError):
                reader.loads(b"not a pickle")
            with pytest.raises(RuntimeError):
                reader.loads(payload)
            with pytest.raises(RuntimeError):
                reader.release_views()
            reader.close()
            reader.close()
        with ProcessArrayReader() as reader:
            np.testing.assert_array_equal(reader.loads(payload), valid)
    with pytest.raises(RuntimeError):
        store.dumps(valid)
    store.close()
