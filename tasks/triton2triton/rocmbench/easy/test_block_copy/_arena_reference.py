"""Protected block-copy oracle and explicit invalid-argument contract.

Triton 4cff872c, language/core.py:_block_ptr.load rejects integer NaN padding.
All values compared here belong to defined output regions; omitted padding has
undefined loaded values beyond N//2, not an implicit zero guarantee.
"""
import torch


class NumericalMismatch(AssertionError):
    pass


INTEGER_NAN_ERROR = 'Padding option `nan` is not supported for integer block pointers'


def compare(actual, expected):
    if not isinstance(actual, torch.Tensor) or (actual.shape != expected.shape or
            actual.dtype != expected.dtype or actual.device != expected.device):
        raise ValueError('Output shape/dtype/device violates the block-copy contract')
    # Byte-exact for copied values/state: no float comparison may flush a
    # subnormal or treat a changed NaN payload as an unchanged input buffer.
    if not torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
        raise NumericalMismatch('Exact block-copy/state mismatch')


def check_output(a, b, source_before, padding, *, untouched_tail=None):
    compare(a, source_before)
    if b.shape != a.shape or b.dtype != a.dtype or b.device != a.device:
        raise ValueError('Output metadata differs from the declared input/output contract')
    half=a.numel()//2
    compare(b[:half], source_before[:half])
    if padding=='zero':
        # Zero sign is not part of the operator contract; nonzero subnormals
        # still fail via integer magnitude bits.
        if b.is_floating_point():
            integer,mask=(torch.int32,0x7fffffff) if b.dtype==torch.float32 else (torch.int16,0x7fff)
            if torch.any((b[half:].contiguous().view(integer)&mask)!=0):
                raise NumericalMismatch('Nonzero padding magnitude')
        elif torch.any(b[half:]!=0):
            raise NumericalMismatch('Nonzero padding')
    elif padding=='nan':
        if not b.is_floating_point() or not torch.isnan(b[half:]).all():
            raise NumericalMismatch('NaN padding missing or replaced by finite/infinite values')
    elif untouched_tail is not None:
        # Performance None launches only N//2: unlike full-grid correctness,
        # the unlaunched suffix must preserve its input sentinel exactly.
        compare(b[half:],untouched_tail)


def expected_nan_error(exc, compilation_error_type):
    if not isinstance(exc,compilation_error_type):return False
    seen=set();cause=exc
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if type(cause) is ValueError and str(cause)==INTEGER_NAN_ERROR:return True
        cause=cause.__cause__
    return False


def expect_integer_nan_rejection(launch,a,b):
    from triton.compiler.errors import CompilationError
    before_a=a.clone();before_b=b.clone()
    try:
        launch()
    except Exception as exc:
        if not expected_nan_error(exc,CompilationError):raise
        compare(a,before_a);compare(b,before_b)
        return {'expected_rejection_checked':True,'exception':'CompilationError',
                'cause_type':'ValueError','cause_message':INTEGER_NAN_ERROR}
    raise AssertionError('Integer NaN-padding invocation unexpectedly succeeded')


class CopyCheck:
    def __init__(self,c):
        self.a,self.b=c['a'],c['b'];self.padding=c['padding_option']
        self.source=self.a.clone();self.destination=self.b.clone()
        self.b.fill_(-7)
        self.tail=self.b[self.a.numel()//2:].clone()

    def check(self,output):
        if not isinstance(output,torch.Tensor) or output.data_ptr()!=self.b.data_ptr():
            raise ValueError('Timed output is not the actual declared destination buffer')
        check_output(self.a,output,self.source,self.padding,
                     untouched_tail=self.tail if self.padding is None else None)

    def fresh(self):
        if self.a.is_floating_point():self.a.copy_(-self.source+0.25)
        else:self.a.copy_(self.source ^ 1)
        self.source_fresh=self.a.clone()
        self.b.fill_(-11)
        self.tail_fresh=self.b[self.a.numel()//2:].clone()

    def check_fresh(self,output):
        if not isinstance(output,torch.Tensor) or output.data_ptr()!=self.b.data_ptr():
            raise ValueError('Replay output is not the actual destination buffer')
        check_output(self.a,output,self.source_fresh,self.padding,
                     untouched_tail=self.tail_fresh if self.padding is None else None)

    def restore(self):
        self.a.copy_(self.source);self.b.copy_(self.destination)


def prepare(c,module):
    return CopyCheck(c)
