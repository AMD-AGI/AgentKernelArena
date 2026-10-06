"""Exact inverse reachability for16 packed BF16 atomic additions on CPU.

Scalar subset minima/maxima are exact because rounded addition is monotone.
Binary search in the full BF16 addition table finds every predecessor within
those bounds. A shared subset walk couples both lanes and returns an order.
"""
from array import array
from bisect import bisect_left,bisect_right

BASE=127  # Ordered BF16 -infinity; +infinity is65408. NaNs are excluded.
END=65408
ZERO=32768

def ordered(bits):return 65535-bits if bits&32768 else bits^32768

def raw_bits(code):return 65535-code if code<32768 else code^32768


class ExactPairOrders:
    def __init__(self,partials):
        import torch
        if len(partials)!=16 or any(len(p)!=2 for p in partials):raise ValueError('Expected16 paired partials')
        if any((b&0x7f80)==0x7f80 for p in partials for b in p):raise ValueError('Nonfinite native partial')
        codes=torch.arange(BASE,END+1,dtype=torch.int32,device='cpu')
        bits=torch.where(codes<32768,65535-codes,codes^32768)
        values=(bits<<16).view(torch.float32)
        tables={}
        for bit in sorted({b for p in partials for b in p}):
            partial=torch.tensor([bit if bit<32768 else bit-65536],dtype=torch.int16,device='cpu').view(torch.bfloat16).float()[0]
            result=(values+partial).to(torch.bfloat16).view(torch.int16).int()&65535
            result=torch.where(result<32768,result^32768,65535-result)
            if not bool((result[1:]>=result[:-1]).all()):raise AssertionError('BF16 addition table is not monotone')
            tables[bit]=array('H',result.tolist())
        self.tables=[(tables[a],tables[b]) for a,b in partials]
        self.low0=array('H',[ZERO])*65536;self.high0=array('H',[ZERO])*65536
        self.low1=array('H',[ZERO])*65536;self.high1=array('H',[ZERO])*65536
        for mask in range(1,65536):
            lo0=lo1=END;hi0=hi1=BASE;remaining=mask
            while remaining:
                bit=remaining&-remaining;j=bit.bit_length()-1;previous=mask^bit;remaining^=bit
                t0,t1=self.tables[j]
                lo0=min(lo0,t0[self.low0[previous]-BASE]);hi0=max(hi0,t0[self.high0[previous]-BASE])
                lo1=min(lo1,t1[self.low1[previous]-BASE]);hi1=max(hi1,t1[self.high1[previous]-BASE])
            self.low0[mask]=lo0;self.high0[mask]=hi0;self.low1[mask]=lo1;self.high1[mask]=hi1
        self.failed=set();self.known={};self.visited=0

    def solve(self,target):
        if len(target)!=2 or any((b&0x7f80)==0x7f80 for b in target):return None
        def visit(mask,c0,c1):
            if not mask:return () if c0==ZERO and c1==ZERO else None
            key=(mask<<32)|(c0<<16)|c1
            if key in self.failed:return None
            if key in self.known:return self.known[key]
            self.visited+=1
            if not (self.low0[mask]<=c0<=self.high0[mask] and self.low1[mask]<=c1<=self.high1[mask]):
                self.failed.add(key);return None
            remaining=mask
            while remaining:
                bit=remaining&-remaining;j=bit.bit_length()-1;previous=mask^bit;remaining^=bit;t0,t1=self.tables[j]
                l0=bisect_left(t0,c0,self.low0[previous]-BASE,self.high0[previous]-BASE+1)
                h0=bisect_right(t0,c0,l0,self.high0[previous]-BASE+1)
                if l0==h0:continue
                l1=bisect_left(t1,c1,self.low1[previous]-BASE,self.high1[previous]-BASE+1)
                h1=bisect_right(t1,c1,l1,self.high1[previous]-BASE+1)
                for a in range(l0,h0):
                    for b in range(l1,h1):
                        prefix=visit(previous,a+BASE,b+BASE)
                        if prefix is not None:
                            result=prefix+(j,);self.known[key]=result;return result
            self.failed.add(key);return None
        result=visit(65535,ordered(target[0]),ordered(target[1]))
        return list(result) if result is not None else None
