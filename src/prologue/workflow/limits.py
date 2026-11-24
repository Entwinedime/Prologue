"""Bound one external operation and monitor storage owned by its collection."""

import os
import subprocess
import time
from dataclasses import dataclass


class AllowanceExhausted(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Usage:
    native_ns: int = 0
    attempts: int = 0
    processing_ns: int = 0

    @classmethod
    def total(cls, records):
        native_ns = attempts = processing_ns = 0
        for record in records:
            native_ns += record.native_ns
            attempts += record.attempts
            processing_ns += record.processing_ns
        return cls(native_ns, attempts, processing_ns)

    def available_to(self, owner, spent):

        # A child resumes on its original charges plus the parent's unused allowance.
        return Usage(owner.native_ns + max(0, self.native_ns - spent.native_ns),
                     owner.attempts + max(0, self.attempts - spent.attempts),
                     owner.processing_ns + max(0, self.processing_ns - spent.processing_ns))

    def document(self):
        return {"native_charged_ns": self.native_ns, "native_attempts": self.attempts, "postprocessing_ns": self.processing_ns}


def retained_bytes(root):
    total = 0
    for directory, _, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except FileNotFoundError:

                # Atomic publication and SDK cleanup can remove a listed file before stat.
                continue
    return total


class OperationLimit:
    def __init__(self, root=None, storage_bytes=None, published_bytes=None):
        self.root, self.storage_bytes = root, storage_bytes
        self.published_bytes = published_bytes
        self.sampled_peak_bytes = None

    def sample(self):
        if self.root is None:
            return None
        size = retained_bytes(self.root)
        if self.published_bytes:
            size += self.published_bytes()
        self.sampled_peak_bytes = max(size, self.sampled_peak_bytes or 0)
        return size

    def check(self):
        size = self.sample()
        if size is not None and self.storage_bytes is not None and size > self.storage_bytes:
            raise AllowanceExhausted("storage allowance exhausted")

    def wait(self, process, deadline):
        timeout = max(0, deadline - time.monotonic())
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            try:
                return process.wait(timeout=min(1, remaining) if self.root is not None else remaining)
            except subprocess.TimeoutExpired:
                self.check()

    def run(self, command, log, timeout):
        self.check()
        deadline = time.monotonic() + timeout
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            return self.wait(process, deadline)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
