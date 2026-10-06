"""Lexical C++ device-body boundary; preprocessing and host source stay frozen."""
import re


def tokens(text, strict=True):
    if strict and re.search(r"\\[ \t]*[\r\n]|\?\?[=/'()!<>-]",text):
        raise ValueError('Line splicing/trigraphs are outside editable device bodies')
    i=0
    while i<len(text):
        if text.startswith('//',i):
            end=text.find('\n',i+2);i=len(text) if end<0 else end+1
        elif text.startswith('/*',i):
            end=text.find('*/',i+2)
            if end<0:raise ValueError('Unterminated C++ comment')
            i=end+2
        elif text.startswith('R"',i):
            opening=text.find('(',i+2);delimiter=text[i+2:opening]
            if opening<0 or len(delimiter)>16 or re.search(r'[\s()\\]',delimiter):raise ValueError('Invalid raw string')
            closing=text.find(')'+delimiter+'"',opening+1)
            if closing<0:raise ValueError('Unterminated raw string')
            i=closing+len(delimiter)+2
        elif text[i] in '\"\'':
            quote=text[i];i+=1
            while i<len(text) and text[i]!=quote:
                if text[i] in '\r\n':raise ValueError('Unterminated C++ literal')
                i+=2 if text[i]=='\\' else 1
            if i>=len(text):raise ValueError('Unterminated C++ literal')
            i+=1
        else:
            if strict and text[i:i+2] in ('<%','%>','%:'):raise ValueError('C++ digraph escape is forbidden')
            yield i,text[i];i+=1


def span(source,marker):
    if source.count(marker)!=1:raise ValueError('Device definition marker missing or duplicated')
    start=source.index('{',source.index(marker)+len(marker))+1;depth=1
    for offset,token in tokens(source[start:],False):
        if token=='{':depth+=1
        elif token=='}':
            depth-=1
            if depth==0:return start,start+offset
    raise ValueError('Unbalanced device body')


def segments(body):
    directives=[];parts=[];position=0;covered=0
    for index,token in tokens(body,False):
        if token!='#' or index<covered:continue
        start=body.rfind('\n',0,index)+1
        if body[start:index].strip():raise ValueError('Preprocessor token inside executable code')
        end=body.find('\n',index)
        if end<0:end=len(body)
        else:end+=1
        while body[start:end].rstrip('\r\n').endswith('\\'):
            following=body.find('\n',end)
            end=len(body) if following<0 else following+1
        parts.append(body[position:start]);directives.append(body[start:end]);position=end;covered=end
    parts.append(body[position:]);return parts,directives


def brace_profile(text):
    depth=0;minimum=0
    for _,token in tokens(text):
        if token=='{':depth+=1
        elif token=='}':depth-=1;minimum=min(minimum,depth)
    return depth,minimum


def check_body(candidate,reference):
    edited,directives=segments(candidate);original,frozen=segments(reference)
    if directives!=frozen:raise ValueError('Device preprocessing directives are frozen')
    for part,old in zip(edited,original):
        before=brace_profile(old);after=brace_profile(part)
        if after[0]!=before[0] or after[1]<before[1]:raise ValueError('Device body changes the enclosing C++ scope')
        code=''.join(token for _,token in tokens(part))
        code=re.sub(r'\busing\s+namespace\s+[A-Za-z_]\w*(?:::\w+)*\s*;','',code)
        code=re.sub(r'__attribute__\s*\(\(\s*ext_vector_type\s*\([0-9]+\)\s*\)\)','',code)
        if re.search(r'\b(?:__host__|__attribute__?|_Pragma|__pragma|constructor|destructor|init_priority|class|struct|namespace|extern)\b',code):
            raise ValueError('Host declarations/attributes are forbidden in device code')


def validate_cpp(candidate,reference,markers):
    edited=[];original=[]
    for marker in markers:
        left=span(candidate,marker);right=span(reference,marker)
        check_body(candidate[left[0]:left[1]],reference[right[0]:right[1]])
        edited.append(left);original.append(right)
    def mask(source,ranges):
        for start,end in sorted(ranges,reverse=True):source=source[:start]+'/*device body*/'+source[end:]
        return source
    if mask(candidate,edited)!=mask(reference,original):raise ValueError('Host code, includes, declarations and device ABI are frozen')
