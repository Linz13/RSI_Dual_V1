"""Deterministic CER/WER/mixed edit rate. Never compare intermediate transcripts."""
import re
import unicodedata
from fractions import Fraction

VERSION = 'content_nfkc_numbers_edit.v6.1'
ONES = dict(zip(('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split(),range(20)))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(),range(20,100,10)))
NUMWORDS = {**ONES, **TENS, 'hundred':100, 'thousand':1000, 'million':1000000}


def _english_number(m):
    words = [w for w in m.group().replace('-', ' ').split() if w != 'and']
    def small(ws):
        if not ws: return 0
        if len(ws)==1 and ws[0] in {**ONES,**TENS}: return NUMWORDS[ws[0]]
        if len(ws)==2 and ws[0] in TENS and ws[1] in ONES and 1<=ONES[ws[1]]<=9:
            return TENS[ws[0]]+ONES[ws[1]]
        if len(ws)>=2 and ws[0] in ONES and 1<=ONES[ws[0]]<=9 and ws[1]=='hundred':
            tail=small(ws[2:])
            if tail<100:return ONES[ws[0]]*100+tail
        raise ValueError('Ambiguous spoken number sequence')
    try:
        total=0;segment=[];last_scale=float('inf')
        for w in words:
            if w in ('thousand','million'):
                scale=NUMWORDS[w]
                if not segment or scale>=last_scale:raise ValueError('Invalid number scale order')
                total+=small(segment)*scale;segment=[];last_scale=scale
            else:segment.append(w)
        return str(total+small(segment))
    except ValueError:
        # "one two three" is not six; leave ambiguous digit/year sequences untouched.
        return m.group()


def _zh_number(m):
    values = dict(zip('零〇一二两三四五六七八九', (0,0,1,2,2,3,4,5,6,7,8,9)))
    s=m.group()
    if not any(x in s for x in '十百千万'): return ''.join(str(values[x]) for x in s)
    total=part=num=0
    for x in s:
        if x in values: num=values[x]
        elif x=='万': total+=(part+num)*10000;part=num=0
        else: part+=(num or 1)*{'十':10,'百':100,'千':1000}[x];num=0
    return str(total+part+num)


def normalize_text(text):
    text=unicodedata.normalize('NFKC',text).casefold()
    # Explicit numeric forms only; keep ordinary Chinese words such as 一会/统一 intact.
    text=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',text)
    alternatives='|'.join(sorted(NUMWORDS,key=len,reverse=True))
    pattern=r'\b(?:'+alternatives+r')(?:[ -]+(?:and[ -]+)?(?:'+alternatives+r'))*\b'
    text=re.sub(pattern,_english_number,text)
    text=re.sub(r'[零〇一二两三四五六七八九十百千万]+(?=年|月|日|号|点|分|秒|个|次|元|岁|米|公里|公斤|小时)',_zh_number,text)
    # Apostrophes, decimal points and signs inside a number retain content meaning.
    text=text.replace('’',"'")
    text=re.sub(r"(?<=[a-z])'(?=[a-z])",'',text)
    text=''.join(ch if not unicodedata.category(ch).startswith(('P','S')) or
                 (ch in '.+-' and i+1<len(text) and text[i+1].isdigit()) else ' '
                 for i,ch in enumerate(text))
    return ' '.join(text.split())


def _tokens(text, mode):
    parts=re.findall(r'[\u3400-\u9fff]|[+-]?\d+(?:\.\d+)?|[^\W\d_]+',text,re.UNICODE)
    if mode=='cer': return [ch for p in parts for ch in p]
    return parts


def content_check(reference, generated, threshold=0.1):
    if not isinstance(reference,str) or not isinstance(generated,str):
        raise ValueError('ASR results must be strings; failure is not an empty transcript')
    a,b=normalize_text(reference),normalize_text(generated)
    mode='mixed' if re.search(r'[\u3400-\u9fff]',a) and re.search('[a-z]',a) else 'cer' if re.search(r'[\u3400-\u9fff]',a) else 'wer'
    aa,bb=_tokens(a,mode),_tokens(b,mode)
    if not aa: raise ValueError('Empty reference content')
    previous=list(range(len(bb)+1))
    for i,x in enumerate(aa,1):
        current=[i]
        for j,y in enumerate(bb,1): current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    edits=previous[-1]
    return {'status':'complete','passed':Fraction(edits,len(aa))<=Fraction(str(threshold)),
            'error_rate':edits/len(aa),'edits':edits,'reference_units':len(aa),'metric':mode,
            'reference_normalized':a,'generated_normalized':b,'version':VERSION}
