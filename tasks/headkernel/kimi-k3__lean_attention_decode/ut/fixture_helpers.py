from array import array
import hashlib
from pathlib import Path
import random

def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''):result.update(part)
    return result.hexdigest()

def checked_path(root, relative, expected=None):
    path=Path(root)/relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Fixture/source must be a regular task-local file: '+str(relative))
    if expected is not None and file_sha(path)!=expected:raise ValueError('Fixture/source hash differs: '+str(relative))
    return path

def scale_offset(row, column, padded_columns):
    """AITER's native 32-row/8-column E8M0 physical byte permutation."""
    return ((row//32)*(padded_columns*32)+(column//8)*256+(column%4)*64
            +(row%16)*4+((column//4)%2)*2+((row//16)%2))

def weighted_work(histogram, rng):
    value=rng.randrange(sum(count for _,count in histogram))
    for rows,count in histogram:
        if value<count:return rows
        value-=count
    raise AssertionError('Invalid observed work histogram')

def make_routes(tokens, tile_m, valid_rows, capacity_rows, expert_slots, seed):
    """Fresh legal top-16 assignments with exactly an observed padded-work count.

    Every expert receives ceil(count/tile_m) sorted blocks, with no artificial
    empty blocks. Cyclic assignment makes each token appear exactly 16 times
    and prevents duplicate experts within a token.
    """
    if valid_rows%tile_m or valid_rows>capacity_rows:raise ValueError('Invalid observed sorted-row extent')
    rng=random.Random(seed); routes=tokens*16; blocks=valid_rows//tile_m
    active=min(896,blocks,routes)
    block_counts=[blocks//active+(i<blocks%active) for i in range(active)]
    counts=[(n-1)*tile_m+1 for n in block_counts]
    remaining=routes-sum(counts)
    if remaining<0 or remaining>active*(tile_m-1):raise ValueError('Observed work count cannot represent the frozen routing')
    order=list(range(active));rng.shuffle(order)
    for position,index in enumerate(order):
        following=len(order)-position-1
        low=max(0,remaining-following*(tile_m-1));high=min(tile_m-1,remaining)
        addition=rng.randint(low,high)
        counts[index]+=addition;remaining-=addition
    if remaining or max(counts)>tokens:raise ValueError('Route degrees do not fit distinct per-token experts')
    expert_ids=list(range(896));rng.shuffle(expert_ids)
    token_order=list(range(tokens));rng.shuffle(token_order)
    topk=array('i',[-1])*(tokens*16);sorted_ids=array('i',[(16<<24)|tokens])*capacity_rows
    sorted_experts=array('i',[-1])*expert_slots;slots=[0]*tokens
    row=0;cursor=0
    for expert,count,block_count in zip(expert_ids,counts,block_counts):
        for j in range(count):
            token=token_order[(cursor+j)%tokens];slot=slots[token];slots[token]+=1
            topk[token*16+slot]=expert
            sorted_ids[row+j]=(slot<<24)|token
        for block in range(block_count):sorted_experts[row//tile_m+block]=expert
        cursor+=count;row+=block_count*tile_m
    if row!=valid_rows or any(n!=16 for n in slots):raise AssertionError('Fresh routing did not preserve exact native work')
    return {'sorted_token_ids':sorted_ids,'sorted_expert_ids':sorted_experts,'topk_ids':topk,
            'num_valid_ids':array('i',[valid_rows,tokens])}

def raw_storage(tensor, torch):
    storage=tensor.untyped_storage()
    return torch.empty(0,dtype=torch.uint8,device=tensor.device).set_(storage,0,(storage.nbytes(),),(1,))
