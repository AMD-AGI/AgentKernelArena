import hashlib
from pathlib import Path
from routing import make_routes

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


def raw_storage(tensor, torch):
    storage=tensor.untyped_storage()
    return torch.empty(0,dtype=torch.uint8,device=tensor.device).set_(storage,0,(storage.nbytes(),),(1,))
