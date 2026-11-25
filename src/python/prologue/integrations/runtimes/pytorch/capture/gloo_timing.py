"""Read the TCP transport boundaries owned by native PyTorch point-to-point calls."""

from ctypes import CDLL, POINTER, Structure, c_int, c_size_t, c_uint64


class GlooSample(Structure):
    _fields_ = [
        ("post_ns", c_uint64), ("completion_begin_ns", c_uint64), ("completion_end_ns", c_uint64),
        ("slot", c_uint64), ("peer", c_int), ("completion_thread", c_int),
    ]

    def document(self):
        if not self.post_ns:
            return None
        completion = [self.completion_begin_ns, self.completion_end_ns] if self.completion_begin_ns else None
        return {"post_raw_ns": self.post_ns, "completion_raw_ns": completion, "slot": self.slot, "peer": self.peer,
                "completion_thread": self.completion_thread if completion is not None else None}


class GlooTiming:
    clock = "monotonic_raw_ns"

    def __init__(self):
        runtime = CDLL(None)
        self.begin_window = runtime.prologue_gloo_begin_window
        self.begin_post = runtime.prologue_gloo_begin_post
        self.end_post = runtime.prologue_gloo_end_post
        self.capture = runtime.prologue_gloo_capture
        self.begin_window.argtypes, self.begin_window.restype = (), None
        self.begin_post.argtypes, self.begin_post.restype = (c_int,), c_uint64
        self.end_post.argtypes, self.end_post.restype = (), None
        self.capture.argtypes, self.capture.restype = (POINTER(c_uint64), c_size_t, POINTER(GlooSample)), None

    def document(self, posts):
        size = len(posts)
        tickets = (c_uint64 * size)(*(post.pop("transport_ticket", 0) for post in posts))
        samples = (GlooSample * size)()
        self.capture(tickets, size, samples)
        for post, sample in zip(posts, samples, strict=True):
            post["transport"] = sample.document()
