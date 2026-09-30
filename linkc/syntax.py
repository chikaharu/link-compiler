"""Lexer and small recursive-descent parser for Rust/Perl style Link source."""
from dataclasses import dataclass
import re
class LinkError(Exception): pass
@dataclass(frozen=True)
class Token:
    kind:str; value:str; pos:int; line:int; col:int
_PATTERN=re.compile(r'(?P<WS>\s+)|(?P<COMMENT>//[^\n]*)|(?P<ARROW>->)|(?P<DCOLON>::)|(?P<NUM>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|(?P<IDENT>\$?[A-Za-z_][A-Za-z_0-9]*)|(?P<SYM>[{}()\[\],;:<>=@+\-†])')
def tokenize(src):
    out=[]; pos=0
    while pos<len(src):
        m=_PATTERN.match(src,pos)
        if not m:
            line=src.count('\n',0,pos)+1; col=pos-src.rfind('\n',0,pos)
            raise LinkError(f'Unexpected character {src[pos]!r} at {line}:{col}')
        if m.lastgroup not in ('WS','COMMENT'):
            line=src.count('\n',0,pos)+1; col=pos-src.rfind('\n',0,pos)
            out.append(Token(m.lastgroup,m.group(),pos,line,col))
        pos=m.end()
    out.append(Token('EOF','',pos,src.count('\n')+1,pos-src.rfind('\n'))); return out
class Parser:
    def __init__(self,src): self.tokens=tokenize(src); self.i=0
    def at(self,v): return self.tokens[self.i].value==v
    def get(self): t=self.tokens[self.i]; self.i+=1; return t
    def expect(self,v):
        t=self.get()
        if t.value!=v: raise LinkError(f'Expected {v!r}, found {t.value or "end of file"!r} at {t.line}:{t.col}')
        return t
    def identifier(self,sigil=False):
        t=self.get()
        if t.kind!='IDENT' or t.value.startswith('$')!=sigil: raise LinkError(f'Expected {"$variable" if sigil else "identifier"}, found {t.value!r} at {t.line}:{t.col}')
        return t.value
    def nat(self):
        t=self.get()
        if t.kind!='NUM' or not t.value.isdigit() or int(t.value)<=0: raise LinkError(f'Expected positive integer at {t.line}:{t.col}')
        return int(t.value)
    def space(self): self.expect('C'); self.expect('['); n=self.nat(); self.expect(']'); return n
    def type_(self):
        self.expect('Link'); self.expect('<'); a=self.space(); self.expect(','); b=self.space(); self.expect('>'); return (a,b)
    def expression(self,min_bp=0):
        t=self.get()
        if t.value=='(': left=self.expression(); self.expect(')')
        elif t.value=='-': left=('neg',self.expression(40))
        elif t.kind=='NUM': left=('num',float(t.value))
        elif t.value=='i': left=('num',1j)
        elif t.kind=='IDENT' and t.value.startswith('$'): left=('var',t.value)
        elif t.value in ('identity','matrix'):
            self.expect('::'); self.expect('<'); a=self.space()
            if t.value=='matrix': self.expect(','); b=self.space()
            else: b=a
            self.expect('>'); self.expect('('); args=self.array() if t.value=='matrix' else None; self.expect(')'); left=(t.value,a,b,args)
        elif t.value=='phase2':
            self.expect('('); arg=self.expression(); self.expect(')'); left=('phase2',arg)
        else: raise LinkError(f'Unexpected expression token {t.value!r} at {t.line}:{t.col}')
        while True:
            if self.at('†') and 50>=min_bp: self.get(); left=('adjoint',left); continue
            op=self.tokens[self.i].value
            if op not in ('@','+','-'): break
            bp=20 if op=='@' else 10
            if bp<min_bp: break
            self.get(); right=self.expression(bp+1); left=({'@':'compose','+':'add','-':'sub'}[op],left,right)
        return left
    def array(self):
        self.expect('['); vals=[]
        if not self.at(']'):
            while True:
                vals.append(self.array() if self.at('[') else self.expression())
                if not self.at(','): break
                self.get()
                if self.at(']'): break
        self.expect(']'); return vals
    def parse(self):
        self.expect('fn'); self.expect('main'); self.expect('('); self.expect(')'); self.expect('->'); rt=self.type_(); self.expect('{'); stmts=[]; returned=False
        while not self.at('}'):
            if self.tokens[self.i].kind=='EOF': raise LinkError('Unterminated function body')
            if self.at('let'):
                self.get(); name=self.identifier(sigil=True); ann=None
                if self.at(':'): self.get(); ann=self.type_()
                self.expect('='); expr=self.expression(); self.expect(';'); stmts.append(('let',name,ann,expr))
            elif self.at('return'):
                self.get(); expr=self.expression(); self.expect(';'); stmts.append(('return',expr)); returned=True; break
            else:
                t=self.tokens[self.i]; raise LinkError(f'Expected let or return at {t.line}:{t.col}')
        self.expect('}')
        if not returned: raise LinkError('main must return a Link')
        if self.tokens[self.i].kind!='EOF': raise LinkError('Unexpected tokens following main')
        return rt,stmts
def parse(src): return Parser(src).parse()
