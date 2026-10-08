"""Own the frozen MLA wrapper's written allocations before either kernel runs.

The module-local torch proxy delegates the original empty/empty_like calls with
unchanged arguments. It does not patch global torch, rewrite the native source,
preallocate buffers, or add work to a captured graph. The BF16 wrapper's one-element
q.new_empty dummy is read-only/unused by the frozen kernels and is not a written
storage; out, m_partial, l_partial and acc_partial are the four written storages.
"""
from functools import update_wrapper


def require(value, message):
    if not value:
        raise RuntimeError('Native write ownership: ' + message)


class WrittenStorages:
    """An invocation's cleanup lease, established before native entry."""
    def __init__(self, tracker):
        self.tracker = tracker
        self.allocations = []
        self._storages = {}
        self.entered = False

    def __enter__(self):
        require(not self.entered and not self.allocations and self.tracker.active is None, 'nested or reused invocation scope')
        self.entered = True
        self.tracker.active = self
        return self

    def __exit__(self, *error):
        # Never issue GPU work here: the surrounding CUDA capture context must
        # unwind first. The caller's finally owns scrub + synchronization.
        require(self.tracker.active is self, 'allocation scope changed during invocation')
        self.tracker.active = None
        self.entered = False

    def record(self, allocator, tensor):
        pointer = tensor.untyped_storage().data_ptr()
        self.allocations.append(allocator)
        self._storages.setdefault(pointer, tensor)
        return tensor

    @property
    def tensors(self):
        return tuple(self._storages.values())

    def validate_return(self, output):
        require(self.allocations == ['empty_like', 'empty', 'empty_like', 'empty']
                and len(self._storages) == 4, 'frozen split/reduce allocation sequence differs')
        require(output.untyped_storage().data_ptr() in self._storages,
                'returned output was not owned before native execution')

    def clear(self):
        from snapshots import raw_storage
        errors = []
        for value in self._storages.values():
            try:
                raw_storage(value).fill_(0xAA)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def release(self):
        require(not self.entered, 'cannot release an active allocation scope')
        self._storages.clear()


class ModuleTorch:
    """Forward only this native module's factory calls through its active lease."""
    def __init__(self, original):
        self.original = original
        self.active = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _allocate(self, name, args, kwargs):
        require(self.active is not None, 'native invocation has no cleanup owner')
        value = getattr(self.original, name)(*args, **kwargs)
        # Registration occurs before the frozen wrapper receives the tensor.
        return self.active.record(name, value)

    def empty(self, *args, **kwargs):
        return self._allocate('empty', args, kwargs)

    def empty_like(self, *args, **kwargs):
        return self._allocate('empty_like', args, kwargs)


class TrackedNative:
    def __init__(self, function, tracker):
        self.function = function
        self.allocation_tracker = tracker
        update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        owner = self.allocation_tracker.active
        require(owner is not None, 'enter a written-storage scope before invoking native code')
        output = self.function(*args, **kwargs)
        owner.validate_return(output)
        return output


def track_native_allocations(function):
    namespace = function.__globals__
    require('torch' in namespace and not isinstance(namespace['torch'], ModuleTorch),
            'expected an unwrapped native module torch binding')
    tracker = ModuleTorch(namespace['torch'])
    namespace['torch'] = tracker
    return TrackedNative(function, tracker)


def owned_outputs(function):
    require(isinstance(function, TrackedNative), 'untracked native callable')
    return WrittenStorages(function.allocation_tracker)
