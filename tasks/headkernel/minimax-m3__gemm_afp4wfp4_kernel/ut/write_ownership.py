"""Own the private native wrapper's writable allocations before kernel entry.

Factory arguments and native allocation order are unchanged for both source
legs. Eager allocations are poisoned immediately. Capture only records and
retains allocations; initialize() poisons them outside capture before replay.
Only this wrapper module's torch binding is proxied, never process-global torch.
"""
from contextlib import contextmanager


def require(condition, message):
    if not condition:
        raise RuntimeError("FP4 write ownership: " + message)


def storage_id(tensor):
    return tensor.untyped_storage().data_ptr()


class WrittenStorages:
    def __init__(self, torch_module, readonly):
        self.torch = torch_module
        self.readonly = {storage_id(value) for value in readonly if value is not None}
        self._tensors = {}
        self._active = False
        self._capturing = False
        self._replay_ready = False

    @property
    def tensors(self):
        return tuple(self._tensors.values())

    def _check_writable(self, value):
        require(storage_id(value) not in self.readonly, "writable allocation/output aliases a read-only input")
        require(value.is_floating_point(), "native written storage must have a floating dtype")

    @contextmanager
    def invocation(self, *, capturing):
        require(not self._active, "nested native allocation scope")
        self._tensors = {}
        self._replay_ready = False
        self._active, self._capturing = True, capturing
        try:
            yield
        finally:
            # No GPU work on scope exit, which can still be inside capture.
            self._active, self._capturing = False, False

    def record(self, value):
        require(self._active, "native allocation has no invocation owner")
        self._check_writable(value)
        self._tensors.setdefault(storage_id(value), value)
        if not self._capturing:
            value.fill_(float("nan"))
        return value

    def validate_shapes(self, expected):
        require([tuple(value.shape) for value in self.tensors] == expected,
                "frozen wrapper writable allocation sequence differs")

    def initialize(self, *outputs):
        require(not self._active and not self.torch.cuda.is_current_stream_capturing(),
                "initialization must be outside native invocation and graph capture")
        values = list(self.tensors)
        values.extend(value for value in outputs if value is not None)
        # Validate the entire set before issuing a write. Distinct views may
        # share a writable storage; initialize each view's defined footprint.
        values = list({id(value): value for value in values}.values())
        for value in values:
            self._check_writable(value)
        for value in values:
            value.fill_(float("nan"))
        self._replay_ready = True

    def before_graph_replay(self):
        require(self._replay_ready, "each graph replay requires fresh writable-storage initialization")
        self._replay_ready = False


class ModuleTorch:
    def __init__(self, original, owner):
        self.original, self.owner = original, owner

    def __getattr__(self, name):
        return getattr(self.original, name)

    def empty(self, *args, **kwargs):
        return self.owner.record(self.original.empty(*args, **kwargs))

    def empty_like(self, *args, **kwargs):
        return self.owner.record(self.original.empty_like(*args, **kwargs))


def track_native_allocations(function, readonly):
    namespace = function.__globals__
    require("torch" in namespace and not isinstance(namespace["torch"], ModuleTorch),
            "expected an unwrapped private native module")
    owner = WrittenStorages(namespace["torch"], readonly)
    namespace["torch"] = ModuleTorch(namespace["torch"], owner)
    return owner
