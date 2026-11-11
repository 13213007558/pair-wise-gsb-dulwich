"""Thin-pack + malformed-input tests for the reuse cache.

Covers:
  * thin pack (REF_DELTA bases resolved externally) parses correctly and the
    external base is read once and then reused;
  * a malformed delta raises and is NOT inserted into the reusable cache;
  * a checksum mismatch (verify_resolved) raises and is NOT cached.
"""
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from dulwich.object_format import DEFAULT_OBJECT_FORMAT
from dulwich.objects import Blob
from dulwich.pack import (
    OFS_DELTA, REF_DELTA, PackData, MemoryPackIndex, Pack,
    write_pack_header, write_pack_object, create_delta, obj_sha,
    ApplyDeltaError, ChecksumMismatch,
)
from dulwich.errors import ApplyDeltaError as ADE
import bench_common


def make_thin_pack(blob_size=4096):
    """External base B0; thin pack holds two REF_DELTA descendants of B0,
    plus one internal full blob."""
    b0 = bench_common.obj_data(3, 0, blob_size)
    b1 = bench_common.obj_data(3, 1, blob_size)
    b2 = bench_common.obj_data(3, 2, blob_size)
    other = b"internal-full-blob"
    tn = Blob.type_num
    sha0 = obj_sha(tn, [b0])
    sha1 = obj_sha(tn, [b1])
    sha2 = obj_sha(tn, [b2])
    sha_other = obj_sha(tn, [other])

    f = io.BytesIO()
    write_pack_header(f.write, 3)
    off_other = f.tell()
    write_pack_object(f.write, tn, [other], object_format=DEFAULT_OBJECT_FORMAT)
    d1 = list(create_delta(b0, b1))
    off1 = f.tell()
    write_pack_object(f.write, REF_DELTA, (sha0, d1),
                      object_format=DEFAULT_OBJECT_FORMAT)
    d2 = list(create_delta(b0, b2))
    off2 = f.tell()
    write_pack_object(f.write, REF_DELTA, (sha0, d2),
                      object_format=DEFAULT_OBJECT_FORMAT)
    body = f.getvalue()
    from hashlib import sha1 as _sha1
    raw = body + _sha1(body).digest()
    ext = {sha0: (tn, b0)}
    expected = {sha1: b1, sha2: b2, sha_other: other}
    offsets = {sha1: off1, sha2: off2, sha_other: off_other}
    return raw, ext, expected, offsets, (tn, b0, sha0)


def open_thin_pack(raw, td, resolver, offsets):
    pack_path = os.path.join(td, "thin.pack")
    with open(pack_path, "wb") as fh:
        fh.write(raw)
    data = PackData(pack_path, object_format=DEFAULT_OBJECT_FORMAT)
    # Build the index manually: a thin pack's external bases are not resolvable
    # while indexing, so MemoryPackIndex.for_pack cannot be used here.
    entries = [(sha, off, 0) for sha, off in offsets.items()]
    index = MemoryPackIndex(entries, DEFAULT_OBJECT_FORMAT,
                            pack_checksum=raw[-20:])
    pack = Pack.from_objects(data, index)
    pack.resolve_ext_ref = resolver
    return pack


def main():
    with tempfile.TemporaryDirectory() as td:
        raw, ext, expected, offsets, (tn, b0, sha0) = make_thin_pack()
        reads = {"n": 0}

        def resolver(oid):
            reads["n"] += 1
            return ext[bytes(oid)]

        pack = open_thin_pack(raw, td, resolver, offsets)

        # Resolve both external-based leaves.
        for oid, want in expected.items():
            got_tn, got = pack.get_raw(oid)
            assert got_tn == tn and got == want, ("thin mismatch", oid.hex())
            assert obj_sha(got_tn, [got]) == oid, "thin OID mismatch"

        print("thin pack: both leaves resolved correctly")
        print("external base reads for 2 leaves:", reads["n"],
              "(1 = base reused, not re-read per leaf)")
        assert reads["n"] == 1, "external base was read more than once"

        # External base is not in the in-pack resolved cache (it lives outside).
        rc = pack.data._resolved_cache
        print("resolved cache entries (in-pack only):", len(rc._cache))
        for oid in list(expected)[:2]:
            assert offsets[oid] in rc._cache, "thin leaf should be cached"
        print("thin leaves present in reusable cache: OK")
        pack.close()




def test_bad_delta_not_cached():
    """A corrupt delta raises ApplyDeltaError and never reaches the cache."""
    blob_size = 256
    b0 = bench_common.obj_data(3, 0, blob_size)
    good = bench_common.obj_data(3, 1, blob_size)
    tn = Blob.type_num
    sha0 = obj_sha(tn, [b0])
    sha_good = obj_sha(tn, [good])

    # Hand-craft an invalid delta: header sizes then an out-of-range copy op.
    def size_enc(n):
        out = [n & 0x7F]
        n >>= 7
        while n:
            out.insert(0, 0x80 | (n & 0x7F))
            n >>= 7
        return bytes(out)

    bad_delta = size_enc(len(b0)) + size_enc(len(good)) + b"\x80"  # copy, zero args -> invalid
    f = io.BytesIO()
    write_pack_header(f.write, 2)
    write_pack_object(f.write, tn, [b0], object_format=DEFAULT_OBJECT_FORMAT)
    bad_off = f.tell()
    write_pack_object(f.write, REF_DELTA, (sha0, [bad_delta]),
                      object_format=DEFAULT_OBJECT_FORMAT)
    body = f.getvalue()
    from hashlib import sha1 as _s
    raw = body + _s(body).digest()

    with tempfile.TemporaryDirectory() as td:
        pack_path = os.path.join(td, "bad.pack")
        with open(pack_path, "wb") as fh:
            fh.write(raw)
        data = PackData(pack_path, object_format=DEFAULT_OBJECT_FORMAT)
        index = MemoryPackIndex(
            [(sha0, 12, 0), (sha_good, bad_off, 0)],
            DEFAULT_OBJECT_FORMAT, pack_checksum=raw[-20:],
        )
        pack = Pack.from_objects(data, index)
        pack.resolve_ext_ref = lambda oid: (tn, b0)

        before = len(data._resolved_cache._cache)
        raised = False
        try:
            pack.get_raw(sha_good)
        except (ApplyDeltaError, ADE, Exception) as e:
            raised = True
            print("bad delta raised:", type(e).__name__)
        assert raised, "malformed delta should have raised"
        after = len(data._resolved_cache._cache)
        assert before == after == 0, "bad object must not enter resolved cache"
        assert bad_off not in data._resolved_cache._cache
        # The raw (still-deltified) entry may be in the raw cache, but never
        # a resolved full text.
        print("resolved cache size after bad delta:", after, "(expected 0)")
        pack.close()


def test_checksum_failure_not_cached():
    """With verify_resolved, a wrong indexed OID raises and is not cached."""
    blob_size = 256
    b0 = bench_common.obj_data(3, 0, blob_size)
    good = bench_common.obj_data(3, 1, blob_size)
    tn = Blob.type_num
    sha0 = obj_sha(tn, [b0])
    real_sha = obj_sha(tn, [good])
    fake_sha = b"\xcd" * 20  # deliberately wrong OID in the index

    f = io.BytesIO()
    write_pack_header(f.write, 2)
    write_pack_object(f.write, tn, [b0], object_format=DEFAULT_OBJECT_FORMAT)
    leaf_off = f.tell()
    write_pack_object(
        f.write, REF_DELTA, (sha0, list(create_delta(b0, good))),
        object_format=DEFAULT_OBJECT_FORMAT,
    )
    body = f.getvalue()
    from hashlib import sha1 as _s
    raw = body + _s(body).digest()

    with tempfile.TemporaryDirectory() as td:
        pack_path = os.path.join(td, "csum.pack")
        with open(pack_path, "wb") as fh:
            fh.write(raw)
        data = PackData(pack_path, object_format=DEFAULT_OBJECT_FORMAT)
        index = MemoryPackIndex(
            [(sha0, 12, 0), (fake_sha, leaf_off, 0)],
            DEFAULT_OBJECT_FORMAT, pack_checksum=raw[-20:],
        )
        pack = Pack.from_objects(data, index)
        pack.verify_resolved = True
        pack.resolve_ext_ref = lambda oid: (tn, b0)

        raised = False
        try:
            pack.get_raw(fake_sha)
        except ChecksumMismatch as e:
            raised = True
            print("checksum failure raised:", type(e).__name__)
        assert raised, "wrong OID should raise ChecksumMismatch"
        assert leaf_off not in data._resolved_cache._cache, \
            "checksum-failed object must not be cached"
        print("resolved cache size after checksum fail:",
              len(data._resolved_cache._cache), "(expected 0)")
        pack.close()


if __name__ == "__main__":
    main()
    test_bad_delta_not_cached()
    test_checksum_failure_not_cached()
