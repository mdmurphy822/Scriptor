//! Chat-template rendering: a Jinja-compatible interpreter covering the constructs used by
//! model `chat_template.jinja` files (blocks, macros, namespaces, loops with `loop.*`, filters,
//! tests, slices and string methods), with `trim_blocks` and `lstrip_blocks` whitespace rules.

use crate::error::{ChatError, Result};
use serde_json::{Map, Value};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(ChatError::Template(message.into()))
}

/// Keyword arguments passed to a template render.
#[derive(Debug, Clone, Default)]
pub struct RenderOptions {
    /// Append the assistant generation prompt.
    pub add_generation_prompt: bool,
    /// Template `enable_thinking`; left undefined when `None`.
    pub enable_thinking: Option<bool>,
    /// Template `reasoning_effort`; left undefined when `None`.
    pub reasoning_effort: Option<String>,
    /// Template `preserve_thinking`; left undefined when `None`.
    pub preserve_thinking: Option<bool>,
    /// Tool declarations; left undefined when `None`.
    pub tools: Option<Value>,
    /// Additional template variables.
    pub extra: Map<String, Value>,
}

impl RenderOptions {
    /// Builds options from a JSON object of template keyword arguments.
    pub fn from_kwargs(kwargs: &Value) -> Result<Self> {
        let Some(map) = kwargs.as_object() else {
            return fail("template kwargs must be an object");
        };
        let mut out = RenderOptions::default();
        for (key, value) in map {
            match key.as_str() {
                "add_generation_prompt" => {
                    out.add_generation_prompt = value.as_bool().unwrap_or(false)
                }
                "enable_thinking" => out.enable_thinking = value.as_bool(),
                "preserve_thinking" => out.preserve_thinking = value.as_bool(),
                "reasoning_effort" => out.reasoning_effort = value.as_str().map(str::to_owned),
                "tools" => out.tools = Some(value.clone()),
                _ => {
                    out.extra.insert(key.clone(), value.clone());
                }
            }
        }
        Ok(out)
    }
}

/// A parsed chat template.
#[derive(Debug)]
pub struct ChatTemplate {
    nodes: Vec<Node>,
}

impl ChatTemplate {
    /// Parses a template from source text.
    pub fn new(source: &str) -> Result<Self> {
        let segments = segment(source)?;
        let mut parser = BlockParser { segments, pos: 0 };
        let (nodes, end) = parser.nodes(&[])?;
        if let Some((kw, _)) = end {
            return fail(format!("unexpected `{kw}`"));
        }
        Ok(Self { nodes })
    }

    /// Parses a template from a file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::new(&std::fs::read_to_string(path)?)
    }

    /// Renders the template for `messages` (a JSON array) with the given options.
    pub fn render(&self, messages: &Value, options: &RenderOptions) -> Result<String> {
        let mut ctx: HashMap<String, V> = HashMap::new();
        for (k, v) in &options.extra {
            ctx.insert(k.clone(), V::from_json(v));
        }
        ctx.insert("messages".into(), V::from_json(messages));
        ctx.insert(
            "add_generation_prompt".into(),
            V::Bool(options.add_generation_prompt),
        );
        if let Some(b) = options.enable_thinking {
            ctx.insert("enable_thinking".into(), V::Bool(b));
        }
        if let Some(b) = options.preserve_thinking {
            ctx.insert("preserve_thinking".into(), V::Bool(b));
        }
        if let Some(s) = &options.reasoning_effort {
            ctx.insert("reasoning_effort".into(), V::Str(s.clone()));
        }
        if let Some(t) = &options.tools {
            ctx.insert("tools".into(), V::from_json(t));
        }
        let mut interp = Interp {
            frames: vec![ctx],
            macros: HashMap::new(),
        };
        let mut out = String::new();
        interp.exec(&self.nodes, &mut out)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// Values

type NsRef = Rc<RefCell<Vec<(String, V)>>>;

#[derive(Debug, Clone)]
enum V {
    Undef,
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<V>),
    Map(Vec<(String, V)>),
    Ns(NsRef),
}

impl V {
    fn from_json(v: &Value) -> V {
        match v {
            Value::Null => V::None,
            Value::Bool(b) => V::Bool(*b),
            Value::Number(n) => match n.as_i64() {
                Some(i) => V::Int(i),
                None => V::Float(n.as_f64().unwrap_or(0.0)),
            },
            Value::String(s) => V::Str(s.clone()),
            Value::Array(a) => V::List(a.iter().map(V::from_json).collect()),
            Value::Object(m) => V::Map(
                m.iter()
                    .map(|(k, v)| (k.clone(), V::from_json(v)))
                    .collect(),
            ),
        }
    }

    fn truthy(&self) -> bool {
        match self {
            V::Undef | V::None => false,
            V::Bool(b) => *b,
            V::Int(i) => *i != 0,
            V::Float(f) => *f != 0.0,
            V::Str(s) => !s.is_empty(),
            V::List(l) => !l.is_empty(),
            V::Map(m) => !m.is_empty(),
            V::Ns(_) => true,
        }
    }

    fn is_iterable(&self) -> bool {
        matches!(self, V::Str(_) | V::List(_) | V::Map(_))
    }

    fn to_text(&self) -> String {
        match self {
            V::Undef => String::new(),
            V::None => "None".into(),
            V::Bool(true) => "True".into(),
            V::Bool(false) => "False".into(),
            V::Int(i) => i.to_string(),
            V::Float(f) => float_repr(*f),
            V::Str(s) => s.clone(),
            other => {
                let mut s = String::new();
                write_json(other, &mut s);
                s
            }
        }
    }
}

fn float_repr(f: f64) -> String {
    if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

fn eq(a: &V, b: &V) -> bool {
    match (a, b) {
        (V::Undef, V::Undef) | (V::None, V::None) => true,
        (V::Bool(x), V::Bool(y)) => x == y,
        (V::Int(x), V::Int(y)) => x == y,
        (V::Int(x), V::Float(y)) | (V::Float(y), V::Int(x)) => (*x as f64) == *y,
        (V::Float(x), V::Float(y)) => x == y,
        (V::Bool(x), V::Int(y)) | (V::Int(y), V::Bool(x)) => i64::from(*x) == *y,
        (V::Str(x), V::Str(y)) => x == y,
        (V::List(x), V::List(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| eq(p, q)),
        (V::Map(x), V::Map(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.iter().any(|(k2, v2)| k == k2 && eq(v, v2)))
        }
        (V::Ns(x), V::Ns(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

fn num(v: &V) -> Option<f64> {
    match v {
        V::Int(i) => Some(*i as f64),
        V::Float(f) => Some(*f),
        V::Bool(b) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

fn compare(a: &V, b: &V) -> Result<Ordering> {
    if let (V::Int(x), V::Int(y)) = (a, b) {
        return Ok(x.cmp(y));
    }
    if let (Some(x), Some(y)) = (num(a), num(b)) {
        return x
            .partial_cmp(&y)
            .map_or_else(|| fail("unordered comparison"), Ok);
    }
    if let (V::Str(x), V::Str(y)) = (a, b) {
        return Ok(x.cmp(y));
    }
    fail("values cannot be ordered")
}

fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Serialises like Python `json.dumps(value, ensure_ascii=False)`.
fn write_json(v: &V, out: &mut String) {
    match v {
        V::Undef | V::None => out.push_str("null"),
        V::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        V::Int(i) => out.push_str(&i.to_string()),
        V::Float(f) => out.push_str(&float_repr(*f)),
        V::Str(s) => write_json_string(s, out),
        V::List(l) => {
            out.push('[');
            for (i, x) in l.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json(x, out);
            }
            out.push(']');
        }
        V::Map(m) => write_json_pairs(m, out),
        V::Ns(n) => write_json_pairs(&n.borrow(), out),
    }
}

fn write_json_pairs(m: &[(String, V)], out: &mut String) {
    out.push('{');
    for (i, (k, x)) in m.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_json_string(k, out);
        out.push_str(": ");
        write_json(x, out);
    }
    out.push('}');
}

// ---------------------------------------------------------------------------------------------
// Template segmentation (whitespace control)

#[derive(Debug)]
enum Segment {
    Data(String),
    Var(String),
    Block(String),
}

/// Index of the closing delimiter `close` at or after `from`, skipping quoted strings.
fn find_close(src: &str, from: usize, close: &str) -> Option<usize> {
    let bytes = src.as_bytes();
    let mut i = from;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let b = bytes[i];
        match quote {
            Some(q) => {
                if b == b'\\' {
                    i += 1;
                } else if b == q {
                    quote = None;
                }
            }
            None => {
                if b == b'\'' || b == b'"' {
                    quote = Some(b);
                } else if src[i..].starts_with(close) {
                    return Some(i);
                }
            }
        }
        i += 1;
    }
    None
}

fn segment(source: &str) -> Result<Vec<Segment>> {
    let src = source
        .strip_suffix('\n')
        .map(|s| s.strip_suffix('\r').unwrap_or(s))
        .unwrap_or(source);
    let mut out = Vec::new();
    let mut pos = 0;
    let mut line_starting = true;
    let mut trim_next = false;
    while pos <= src.len() {
        let rest = &src[pos..];
        let next = ["{{", "{%", "{#"]
            .iter()
            .filter_map(|o| rest.find(o).map(|i| (i, *o)))
            .min_by_key(|(i, _)| *i);
        let (data_end, open) = match next {
            Some((i, o)) => (pos + i, Some(o)),
            None => (src.len(), None),
        };
        let mut data = &src[pos..data_end];
        if trim_next {
            data = data.trim_start();
        }
        let Some(open) = open else {
            if !data.is_empty() {
                out.push(Segment::Data(data.to_owned()));
            }
            break;
        };
        let mut body_start = data_end + 2;
        let minus_left = src[body_start..].starts_with('-');
        if minus_left {
            body_start += 1;
            data = data.trim_end();
        } else if open != "{{" {
            let l_pos = data.rfind('\n').map_or(0, |i| i + 1);
            if (l_pos > 0 || line_starting) && data[l_pos..].chars().all(|c| c == ' ' || c == '\t')
            {
                data = &data[..l_pos];
            }
        }
        if !data.is_empty() {
            out.push(Segment::Data(data.to_owned()));
        }
        let close = match open {
            "{{" => "}}",
            "{%" => "%}",
            _ => "#}",
        };
        let Some(end) = find_close(src, body_start, close) else {
            return fail(format!("unterminated `{open}` tag"));
        };
        let mut content = &src[body_start..end];
        let minus_right = content.ends_with('-');
        if minus_right {
            content = &content[..content.len() - 1];
        }
        pos = end + 2;
        trim_next = minus_right;
        line_starting = false;
        match open {
            "{{" => out.push(Segment::Var(content.to_owned())),
            "{%" => {
                out.push(Segment::Block(content.trim().to_owned()));
                if !minus_right {
                    if src[pos..].starts_with("\r\n") {
                        pos += 2;
                        line_starting = true;
                    } else if src[pos..].starts_with('\n') {
                        pos += 1;
                        line_starting = true;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Expressions

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Name(String),
    Int(i64),
    Float(f64),
    Str(String),
    Op(&'static str),
}

const OPS: [&str; 22] = [
    "==", "!=", "<=", ">=", "//", "**", "+", "-", "*", "/", "%", "~", "<", ">", "=", "(", ")", "[",
    "]", ",", ":", ".",
];

fn unescape(raw: &str) -> Result<String> {
    let mut out = String::new();
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some('\\') => out.push('\\'),
            Some('\'') => out.push('\''),
            Some('"') => out.push('"'),
            Some(k @ ('x' | 'u')) => {
                let n = if k == 'x' { 2 } else { 4 };
                let hex: String = chars.by_ref().take(n).collect();
                let code = u32::from_str_radix(&hex, 16)
                    .ok()
                    .and_then(char::from_u32)
                    .ok_or_else(|| ChatError::Template(format!("bad escape \\{k}{hex}")))?;
                out.push(code);
            }
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    Ok(out)
}

fn lex_expr(src: &str) -> Result<Vec<Tok>> {
    let chars: Vec<char> = src.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '\'' || c == '"' {
            let mut j = i + 1;
            while j < chars.len() && chars[j] != c {
                if chars[j] == '\\' {
                    j += 1;
                }
                j += 1;
            }
            if j >= chars.len() {
                return fail("unterminated string literal");
            }
            let raw: String = chars[i + 1..j].iter().collect();
            toks.push(Tok::Str(unescape(&raw)?));
            i = j + 1;
        } else if c.is_ascii_digit() {
            let mut j = i;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j + 1 < chars.len() && chars[j] == '.' && chars[j + 1].is_ascii_digit() {
                j += 1;
                while j < chars.len() && chars[j].is_ascii_digit() {
                    j += 1;
                }
                let text: String = chars[i..j].iter().collect();
                toks.push(Tok::Float(
                    text.parse()
                        .map_err(|_| ChatError::Template(format!("bad number {text}")))?,
                ));
            } else {
                let text: String = chars[i..j].iter().collect();
                toks.push(Tok::Int(
                    text.parse()
                        .map_err(|_| ChatError::Template(format!("bad number {text}")))?,
                ));
            }
            i = j;
        } else if c.is_alphabetic() || c == '_' {
            let mut j = i;
            while j < chars.len() && (chars[j].is_alphanumeric() || chars[j] == '_') {
                j += 1;
            }
            toks.push(Tok::Name(chars[i..j].iter().collect()));
            i = j;
        } else if c == '|' {
            toks.push(Tok::Op("|"));
            i += 1;
        } else {
            let rest: String = chars[i..chars.len().min(i + 2)].iter().collect();
            match OPS.iter().find(|o| rest.starts_with(**o)) {
                Some(op) => {
                    toks.push(Tok::Op(op));
                    i += op.len();
                }
                None => return fail(format!("unexpected character `{c}` in expression")),
            }
        }
    }
    Ok(toks)
}

#[derive(Debug, Clone)]
enum Expr {
    Lit(V),
    Name(String),
    Attr(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Slice(Box<Expr>, [Option<Box<Expr>>; 3]),
    Call(Box<Expr>, Vec<Expr>, Vec<(String, Expr)>),
    Filter(Box<Expr>, String, Vec<Expr>),
    Test(Box<Expr>, String, Vec<Expr>, bool),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Bin(&'static str, Box<Expr>, Box<Expr>),
    In(Box<Expr>, Box<Expr>, bool),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Cond(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    Seq(Vec<Expr>),
}

type CallArgs = (Vec<Expr>, Vec<(String, Expr)>);

struct ExprParser {
    toks: Vec<Tok>,
    pos: usize,
}

impl ExprParser {
    fn new(src: &str) -> Result<Self> {
        Ok(Self {
            toks: lex_expr(src)?,
            pos: 0,
        })
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn peek_at(&self, n: usize) -> Option<&Tok> {
        self.toks.get(self.pos + n)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    fn at_op(&self, op: &str) -> bool {
        matches!(self.peek(), Some(Tok::Op(o)) if *o == op)
    }

    fn at_name(&self, name: &str) -> bool {
        matches!(self.peek(), Some(Tok::Name(n)) if n == name)
    }

    fn eat_op(&mut self, op: &str) -> bool {
        let hit = self.at_op(op);
        if hit {
            self.pos += 1;
        }
        hit
    }

    fn expect_op(&mut self, op: &str) -> Result<()> {
        if self.eat_op(op) {
            Ok(())
        } else {
            fail(format!("expected `{op}`, found {:?}", self.peek()))
        }
    }

    fn expect_name(&mut self) -> Result<String> {
        match self.next() {
            Some(Tok::Name(n)) => Ok(n),
            other => fail(format!("expected a name, found {other:?}")),
        }
    }

    fn finish(&self) -> Result<()> {
        match self.peek() {
            None => Ok(()),
            Some(t) => fail(format!("unexpected token {t:?}")),
        }
    }

    fn expr(&mut self) -> Result<Expr> {
        let value = self.or()?;
        if self.at_name("if") {
            self.pos += 1;
            let cond = self.or()?;
            let other = if self.at_name("else") {
                self.pos += 1;
                Some(Box::new(self.expr()?))
            } else {
                None
            };
            return Ok(Expr::Cond(Box::new(cond), Box::new(value), other));
        }
        Ok(value)
    }

    fn or(&mut self) -> Result<Expr> {
        let mut l = self.and()?;
        while self.at_name("or") {
            self.pos += 1;
            l = Expr::Or(Box::new(l), Box::new(self.and()?));
        }
        Ok(l)
    }

    fn and(&mut self) -> Result<Expr> {
        let mut l = self.not()?;
        while self.at_name("and") {
            self.pos += 1;
            l = Expr::And(Box::new(l), Box::new(self.not()?));
        }
        Ok(l)
    }

    fn not(&mut self) -> Result<Expr> {
        if self.at_name("not") {
            self.pos += 1;
            return Ok(Expr::Not(Box::new(self.not()?)));
        }
        self.compare()
    }

    fn compare(&mut self) -> Result<Expr> {
        let mut l = self.add()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op(o)) if ["==", "!=", "<", ">", "<=", ">="].contains(o) => Some(*o),
                _ => None,
            };
            if let Some(op) = op {
                self.pos += 1;
                l = Expr::Bin(op, Box::new(l), Box::new(self.add()?));
            } else if self.at_name("in") {
                self.pos += 1;
                l = Expr::In(Box::new(l), Box::new(self.add()?), false);
            } else if self.at_name("not")
                && matches!(self.peek_at(1), Some(Tok::Name(n)) if n == "in")
            {
                self.pos += 2;
                l = Expr::In(Box::new(l), Box::new(self.add()?), true);
            } else {
                return Ok(l);
            }
        }
    }

    fn add(&mut self) -> Result<Expr> {
        let mut l = self.concat()?;
        loop {
            let op = if self.at_op("+") {
                "+"
            } else if self.at_op("-") {
                "-"
            } else {
                return Ok(l);
            };
            self.pos += 1;
            l = Expr::Bin(op, Box::new(l), Box::new(self.concat()?));
        }
    }

    fn concat(&mut self) -> Result<Expr> {
        let mut l = self.mul()?;
        while self.eat_op("~") {
            l = Expr::Bin("~", Box::new(l), Box::new(self.mul()?));
        }
        Ok(l)
    }

    fn mul(&mut self) -> Result<Expr> {
        let mut l = self.unary()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op(o)) if ["*", "/", "//", "%"].contains(o) => *o,
                _ => return Ok(l),
            };
            self.pos += 1;
            l = Expr::Bin(op, Box::new(l), Box::new(self.unary()?));
        }
    }

    fn unary(&mut self) -> Result<Expr> {
        if self.eat_op("-") {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        if self.eat_op("+") {
            return self.unary();
        }
        let primary = self.primary()?;
        let mut node = self.postfix(primary)?;
        loop {
            if self.eat_op("|") {
                let name = self.expect_name()?;
                let args = if self.at_op("(") {
                    self.call_args()?.0
                } else {
                    Vec::new()
                };
                node = Expr::Filter(Box::new(node), name, args);
            } else if self.at_name("is") {
                self.pos += 1;
                let negated = if self.at_name("not") {
                    self.pos += 1;
                    true
                } else {
                    false
                };
                let name = self.expect_name()?;
                let args = if self.at_op("(") {
                    self.call_args()?.0
                } else {
                    Vec::new()
                };
                node = Expr::Test(Box::new(node), name, args, negated);
            } else {
                return Ok(node);
            }
        }
    }

    fn primary(&mut self) -> Result<Expr> {
        match self.next() {
            Some(Tok::Int(i)) => Ok(Expr::Lit(V::Int(i))),
            Some(Tok::Float(f)) => Ok(Expr::Lit(V::Float(f))),
            Some(Tok::Str(mut s)) => {
                while let Some(Tok::Str(more)) = self.peek() {
                    s.push_str(more);
                    self.pos += 1;
                }
                Ok(Expr::Lit(V::Str(s)))
            }
            Some(Tok::Name(n)) => Ok(match n.as_str() {
                "true" | "True" => Expr::Lit(V::Bool(true)),
                "false" | "False" => Expr::Lit(V::Bool(false)),
                "none" | "None" => Expr::Lit(V::None),
                _ => Expr::Name(n),
            }),
            Some(Tok::Op("(")) => {
                if self.eat_op(")") {
                    return Ok(Expr::Seq(Vec::new()));
                }
                let first = self.expr()?;
                if self.at_op(",") {
                    let mut items = vec![first];
                    while self.eat_op(",") {
                        if self.at_op(")") {
                            break;
                        }
                        items.push(self.expr()?);
                    }
                    self.expect_op(")")?;
                    Ok(Expr::Seq(items))
                } else {
                    self.expect_op(")")?;
                    Ok(first)
                }
            }
            Some(Tok::Op("[")) => {
                let mut items = Vec::new();
                while !self.at_op("]") {
                    items.push(self.expr()?);
                    if !self.eat_op(",") {
                        break;
                    }
                }
                self.expect_op("]")?;
                Ok(Expr::Seq(items))
            }
            other => fail(format!("unexpected token {other:?}")),
        }
    }

    fn call_args(&mut self) -> Result<CallArgs> {
        self.expect_op("(")?;
        let mut args = Vec::new();
        let mut kwargs = Vec::new();
        while !self.at_op(")") {
            if let (Some(Tok::Name(n)), Some(Tok::Op("="))) = (self.peek(), self.peek_at(1)) {
                let name = n.clone();
                self.pos += 2;
                kwargs.push((name, self.expr()?));
            } else {
                args.push(self.expr()?);
            }
            if !self.eat_op(",") {
                break;
            }
        }
        self.expect_op(")")?;
        Ok((args, kwargs))
    }

    fn postfix(&mut self, mut node: Expr) -> Result<Expr> {
        loop {
            if self.eat_op(".") {
                match self.next() {
                    Some(Tok::Name(n)) => node = Expr::Attr(Box::new(node), n),
                    Some(Tok::Int(i)) => {
                        node = Expr::Index(Box::new(node), Box::new(Expr::Lit(V::Int(i))))
                    }
                    other => return fail(format!("bad attribute {other:?}")),
                }
            } else if self.at_op("[") {
                self.pos += 1;
                node = self.subscript(node)?;
            } else if self.at_op("(") {
                let (args, kwargs) = self.call_args()?;
                node = Expr::Call(Box::new(node), args, kwargs);
            } else {
                return Ok(node);
            }
        }
    }

    fn subscript(&mut self, base: Expr) -> Result<Expr> {
        let mut parts: [Option<Box<Expr>>; 3] = [None, None, None];
        let mut slot = 0;
        let mut sliced = false;
        loop {
            if self.at_op(":") {
                self.pos += 1;
                sliced = true;
                slot += 1;
                if slot > 2 {
                    return fail("too many `:` in slice");
                }
            } else if self.at_op("]") {
                break;
            } else {
                parts[slot] = Some(Box::new(self.expr()?));
                if !self.at_op(":") && !self.at_op("]") {
                    return fail("bad subscript");
                }
            }
        }
        self.expect_op("]")?;
        if sliced {
            Ok(Expr::Slice(Box::new(base), parts))
        } else {
            match parts[0].take() {
                Some(i) => Ok(Expr::Index(Box::new(base), i)),
                None => fail("empty subscript"),
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Statements

#[derive(Debug)]
struct MacroDef {
    name: String,
    params: Vec<(String, Option<Expr>)>,
    body: Vec<Node>,
}

#[derive(Debug)]
enum Node {
    Data(String),
    Out(Expr),
    Set(Vec<String>, Option<String>, Expr),
    If(Vec<(Expr, Vec<Node>)>, Vec<Node>),
    For(Vec<String>, Expr, Vec<Node>, Vec<Node>),
    Macro(Rc<MacroDef>),
}

struct BlockParser {
    segments: Vec<Segment>,
    pos: usize,
}

impl BlockParser {
    /// Parses nodes until a block whose keyword is in `stops`; returns that block's keyword and rest.
    #[allow(clippy::type_complexity)]
    fn nodes(&mut self, stops: &[&str]) -> Result<(Vec<Node>, Option<(String, String)>)> {
        let mut out = Vec::new();
        while self.pos < self.segments.len() {
            let seg = std::mem::replace(&mut self.segments[self.pos], Segment::Data(String::new()));
            self.pos += 1;
            match seg {
                Segment::Data(d) => out.push(Node::Data(d)),
                Segment::Var(src) => {
                    let mut p = ExprParser::new(&src)?;
                    let e = p.expr()?;
                    p.finish()?;
                    out.push(Node::Out(e));
                }
                Segment::Block(src) => {
                    let (kw, rest) = match src.split_once(char::is_whitespace) {
                        Some((k, r)) => (k.to_owned(), r.trim().to_owned()),
                        None => (src.clone(), String::new()),
                    };
                    if stops.contains(&kw.as_str()) {
                        return Ok((out, Some((kw, rest))));
                    }
                    out.push(self.statement(&kw, &rest)?);
                }
            }
        }
        if stops.is_empty() {
            Ok((out, None))
        } else {
            fail(format!("missing `{}`", stops[stops.len() - 1]))
        }
    }

    fn statement(&mut self, kw: &str, rest: &str) -> Result<Node> {
        match kw {
            "set" => {
                let mut p = ExprParser::new(rest)?;
                let mut names = vec![p.expect_name()?];
                let mut attr = None;
                if p.eat_op(".") {
                    attr = Some(p.expect_name()?);
                } else {
                    while p.eat_op(",") {
                        names.push(p.expect_name()?);
                    }
                }
                p.expect_op("=")?;
                let e = p.expr()?;
                p.finish()?;
                Ok(Node::Set(names, attr, e))
            }
            "if" => {
                let mut branches = Vec::new();
                let mut cond = parse_whole(rest)?;
                let mut otherwise = Vec::new();
                loop {
                    let (body, end) = self.nodes(&["elif", "else", "endif"])?;
                    branches.push((cond, body));
                    match end {
                        Some((k, r)) if k == "elif" => cond = parse_whole(&r)?,
                        Some((k, _)) if k == "else" => {
                            let (body, end) = self.nodes(&["endif"])?;
                            debug_assert!(end.is_some());
                            otherwise = body;
                            break;
                        }
                        _ => break,
                    }
                }
                Ok(Node::If(branches, otherwise))
            }
            "for" => {
                let mut p = ExprParser::new(rest)?;
                let mut targets = vec![p.expect_name()?];
                while p.eat_op(",") {
                    targets.push(p.expect_name()?);
                }
                if !p.at_name("in") {
                    return fail("expected `in` in for statement");
                }
                p.pos += 1;
                let iter = p.expr()?;
                p.finish()?;
                let (body, end) = self.nodes(&["else", "endfor"])?;
                let otherwise = match end {
                    Some((k, _)) if k == "else" => self.nodes(&["endfor"])?.0,
                    _ => Vec::new(),
                };
                Ok(Node::For(targets, iter, body, otherwise))
            }
            "macro" => {
                let mut p = ExprParser::new(rest)?;
                let name = p.expect_name()?;
                p.expect_op("(")?;
                let mut params = Vec::new();
                while !p.at_op(")") {
                    let pname = p.expect_name()?;
                    let default = if p.eat_op("=") { Some(p.expr()?) } else { None };
                    params.push((pname, default));
                    if !p.eat_op(",") {
                        break;
                    }
                }
                p.expect_op(")")?;
                p.finish()?;
                let (body, _) = self.nodes(&["endmacro"])?;
                Ok(Node::Macro(Rc::new(MacroDef { name, params, body })))
            }
            other => fail(format!("unknown statement `{other}`")),
        }
    }
}

fn parse_whole(src: &str) -> Result<Expr> {
    let mut p = ExprParser::new(src)?;
    let e = p.expr()?;
    p.finish()?;
    Ok(e)
}

// ---------------------------------------------------------------------------------------------
// Evaluation

struct Interp {
    frames: Vec<HashMap<String, V>>,
    macros: HashMap<String, Rc<MacroDef>>,
}

fn slice_indices(len: usize, start: Option<i64>, stop: Option<i64>, step: i64) -> Vec<usize> {
    let n = len as i64;
    let norm = |v: i64, lo: i64, hi: i64| -> i64 {
        let v = if v < 0 { v + n } else { v };
        v.clamp(lo, hi)
    };
    let mut out = Vec::new();
    if step > 0 {
        let mut i = start.map_or(0, |s| norm(s, 0, n));
        let end = stop.map_or(n, |s| norm(s, 0, n));
        while i < end {
            out.push(i as usize);
            i += step;
        }
    } else {
        let mut i = start.map_or(n - 1, |s| norm(s, -1, n - 1));
        let end = stop.map_or(-1, |s| norm(s, -1, n - 1));
        while i > end {
            out.push(i as usize);
            i += step;
        }
    }
    out
}

fn as_int(v: &V) -> Result<i64> {
    match v {
        V::Int(i) => Ok(*i),
        V::Bool(b) => Ok(i64::from(*b)),
        other => fail(format!("expected an integer, found {other:?}")),
    }
}

impl Interp {
    fn lookup(&self, name: &str) -> V {
        self.frames
            .iter()
            .rev()
            .find_map(|f| f.get(name))
            .cloned()
            .unwrap_or(V::Undef)
    }

    fn exec(&mut self, nodes: &[Node], out: &mut String) -> Result<()> {
        for node in nodes {
            match node {
                Node::Data(d) => out.push_str(d),
                Node::Out(e) => {
                    let v = self.eval(e)?;
                    out.push_str(&v.to_text());
                }
                Node::Set(names, attr, e) => {
                    let value = self.eval(e)?;
                    self.assign(names, attr.as_deref(), value)?;
                }
                Node::If(branches, otherwise) => {
                    let mut taken = false;
                    for (cond, body) in branches {
                        if self.eval(cond)?.truthy() {
                            self.exec(body, out)?;
                            taken = true;
                            break;
                        }
                    }
                    if !taken {
                        self.exec(otherwise, out)?;
                    }
                }
                Node::For(targets, iter, body, otherwise) => {
                    let iterable = self.eval(iter)?;
                    let items = self.iterate(&iterable)?;
                    if items.is_empty() {
                        self.exec(otherwise, out)?;
                    }
                    let len = items.len();
                    for (i, item) in items.iter().enumerate() {
                        let mut frame = HashMap::new();
                        if targets.len() == 1 {
                            frame.insert(targets[0].clone(), item.clone());
                        } else {
                            let V::List(parts) = item else {
                                return fail("cannot unpack a non-sequence in for");
                            };
                            if parts.len() != targets.len() {
                                return fail("wrong number of values to unpack in for");
                            }
                            for (t, p) in targets.iter().zip(parts) {
                                frame.insert(t.clone(), p.clone());
                            }
                        }
                        let mut lp: Vec<(String, V)> = vec![
                            ("index0".into(), V::Int(i as i64)),
                            ("index".into(), V::Int(i as i64 + 1)),
                            ("revindex0".into(), V::Int((len - i - 1) as i64)),
                            ("revindex".into(), V::Int((len - i) as i64)),
                            ("first".into(), V::Bool(i == 0)),
                            ("last".into(), V::Bool(i + 1 == len)),
                            ("length".into(), V::Int(len as i64)),
                        ];
                        if i > 0 {
                            lp.push(("previtem".into(), items[i - 1].clone()));
                        }
                        if i + 1 < len {
                            lp.push(("nextitem".into(), items[i + 1].clone()));
                        }
                        frame.insert("loop".into(), V::Map(lp));
                        self.frames.push(frame);
                        let r = self.exec(body, out);
                        self.frames.pop();
                        r?;
                    }
                }
                Node::Macro(def) => {
                    self.macros.insert(def.name.clone(), Rc::clone(def));
                }
            }
        }
        Ok(())
    }

    fn assign(&mut self, names: &[String], attr: Option<&str>, value: V) -> Result<()> {
        if let Some(attr) = attr {
            return match self.lookup(&names[0]) {
                V::Ns(ns) => {
                    let mut ns = ns.borrow_mut();
                    match ns.iter_mut().find(|(k, _)| k == attr) {
                        Some(slot) => slot.1 = value,
                        None => ns.push((attr.to_owned(), value)),
                    }
                    Ok(())
                }
                _ => fail(format!(
                    "cannot assign attribute `{attr}` on a non-namespace"
                )),
            };
        }
        let frame = self.frames.last_mut().expect("a frame always exists");
        if names.len() == 1 {
            frame.insert(names[0].clone(), value);
            return Ok(());
        }
        let V::List(parts) = value else {
            return fail("cannot unpack a non-sequence in set");
        };
        if parts.len() != names.len() {
            return fail("wrong number of values to unpack in set");
        }
        for (n, p) in names.iter().zip(parts) {
            frame.insert(n.clone(), p);
        }
        Ok(())
    }

    fn iterate(&self, v: &V) -> Result<Vec<V>> {
        match v {
            V::List(l) => Ok(l.clone()),
            V::Map(m) => Ok(m.iter().map(|(k, _)| V::Str(k.clone())).collect()),
            V::Str(s) => Ok(s.chars().map(|c| V::Str(c.to_string())).collect()),
            V::Undef | V::None => Ok(Vec::new()),
            other => fail(format!("{other:?} is not iterable")),
        }
    }

    fn eval(&mut self, e: &Expr) -> Result<V> {
        match e {
            Expr::Lit(v) => Ok(v.clone()),
            Expr::Name(n) => Ok(self.lookup(n)),
            Expr::Seq(items) => Ok(V::List(
                items
                    .iter()
                    .map(|i| self.eval(i))
                    .collect::<Result<Vec<_>>>()?,
            )),
            Expr::Attr(base, name) => {
                let b = self.eval(base)?;
                getattr(&b, name)
            }
            Expr::Index(base, index) => {
                let b = self.eval(base)?;
                let i = self.eval(index)?;
                getitem(&b, &i)
            }
            Expr::Slice(base, parts) => {
                let b = self.eval(base)?;
                let mut bounds = [None, None, None];
                for (slot, p) in bounds.iter_mut().zip(parts) {
                    if let Some(p) = p {
                        let v = self.eval(p)?;
                        if !matches!(v, V::None) {
                            *slot = Some(as_int(&v)?);
                        }
                    }
                }
                let step = bounds[2].unwrap_or(1);
                if step == 0 {
                    return fail("slice step cannot be zero");
                }
                match b {
                    V::List(l) => Ok(V::List(
                        slice_indices(l.len(), bounds[0], bounds[1], step)
                            .into_iter()
                            .map(|i| l[i].clone())
                            .collect(),
                    )),
                    V::Str(s) => {
                        let chars: Vec<char> = s.chars().collect();
                        Ok(V::Str(
                            slice_indices(chars.len(), bounds[0], bounds[1], step)
                                .into_iter()
                                .map(|i| chars[i])
                                .collect(),
                        ))
                    }
                    other => fail(format!("cannot slice {other:?}")),
                }
            }
            Expr::Neg(inner) => match self.eval(inner)? {
                V::Int(i) => Ok(V::Int(-i)),
                V::Float(f) => Ok(V::Float(-f)),
                other => fail(format!("cannot negate {other:?}")),
            },
            Expr::Not(inner) => Ok(V::Bool(!self.eval(inner)?.truthy())),
            Expr::And(a, b) => {
                let l = self.eval(a)?;
                if l.truthy() { self.eval(b) } else { Ok(l) }
            }
            Expr::Or(a, b) => {
                let l = self.eval(a)?;
                if l.truthy() { Ok(l) } else { self.eval(b) }
            }
            Expr::Cond(cond, a, b) => {
                if self.eval(cond)?.truthy() {
                    self.eval(a)
                } else if let Some(b) = b {
                    self.eval(b)
                } else {
                    Ok(V::Undef)
                }
            }
            Expr::In(item, container, negated) => {
                let i = self.eval(item)?;
                let c = self.eval(container)?;
                Ok(V::Bool(contains(&c, &i)? != *negated))
            }
            Expr::Bin(op, a, b) => {
                let l = self.eval(a)?;
                let r = self.eval(b)?;
                binary(op, &l, &r)
            }
            Expr::Filter(inner, name, args) => {
                let v = self.eval(inner)?;
                let args = args
                    .iter()
                    .map(|a| self.eval(a))
                    .collect::<Result<Vec<_>>>()?;
                filter(name, v, &args)
            }
            Expr::Test(inner, name, args, negated) => {
                let v = self.eval(inner)?;
                let args = args
                    .iter()
                    .map(|a| self.eval(a))
                    .collect::<Result<Vec<_>>>()?;
                Ok(V::Bool(test(name, &v, &args)? != *negated))
            }
            Expr::Call(callee, args, kwargs) => self.call(callee, args, kwargs),
        }
    }

    fn call(&mut self, callee: &Expr, args: &[Expr], kwargs: &[(String, Expr)]) -> Result<V> {
        let argv = args
            .iter()
            .map(|a| self.eval(a))
            .collect::<Result<Vec<_>>>()?;
        let kwv = kwargs
            .iter()
            .map(|(k, a)| Ok((k.clone(), self.eval(a)?)))
            .collect::<Result<Vec<_>>>()?;
        match callee {
            Expr::Name(name) => match name.as_str() {
                "namespace" => Ok(V::Ns(Rc::new(RefCell::new(kwv)))),
                "raise_exception" => fail(argv.first().map(V::to_text).unwrap_or_default()),
                _ => {
                    let Some(def) = self.macros.get(name).cloned() else {
                        return fail(format!("`{name}` is not callable"));
                    };
                    self.call_macro(&def, argv, kwv)
                }
            },
            Expr::Attr(base, method) => {
                let target = self.eval(base)?;
                call_method(&target, method, &argv)
            }
            _ => fail("expression is not callable"),
        }
    }

    fn call_macro(&mut self, def: &MacroDef, argv: Vec<V>, kwv: Vec<(String, V)>) -> Result<V> {
        let mut frame = HashMap::new();
        for (i, (pname, default)) in def.params.iter().enumerate() {
            let value = if let Some(v) = argv.get(i) {
                v.clone()
            } else if let Some((_, v)) = kwv.iter().find(|(k, _)| k == pname) {
                v.clone()
            } else if let Some(d) = default {
                self.eval(d)?
            } else {
                V::Undef
            };
            frame.insert(pname.clone(), value);
        }
        self.frames.push(frame);
        let mut out = String::new();
        let r = self.exec(&def.body, &mut out);
        self.frames.pop();
        r?;
        Ok(V::Str(out))
    }
}

fn getattr(base: &V, name: &str) -> Result<V> {
    match base {
        V::Map(m) => Ok(m
            .iter()
            .find(|(k, _)| k == name)
            .map_or(V::Undef, |(_, v)| v.clone())),
        V::Ns(n) => Ok(n
            .borrow()
            .iter()
            .find(|(k, _)| k == name)
            .map_or(V::Undef, |(_, v)| v.clone())),
        V::Undef => fail(format!("`{name}` accessed on an undefined value")),
        _ => Ok(V::Undef),
    }
}

fn getitem(base: &V, index: &V) -> Result<V> {
    match (base, index) {
        (V::List(l), V::Int(i)) => {
            let idx = if *i < 0 { *i + l.len() as i64 } else { *i };
            Ok(usize::try_from(idx)
                .ok()
                .and_then(|i| l.get(i))
                .cloned()
                .unwrap_or(V::Undef))
        }
        (V::Str(s), V::Int(i)) => {
            let chars: Vec<char> = s.chars().collect();
            let idx = if *i < 0 { *i + chars.len() as i64 } else { *i };
            Ok(usize::try_from(idx)
                .ok()
                .and_then(|i| chars.get(i))
                .map_or(V::Undef, |c| V::Str(c.to_string())))
        }
        (V::Map(_) | V::Ns(_), V::Str(k)) => getattr(base, k),
        (V::Undef, _) => fail("subscript on an undefined value"),
        _ => Ok(V::Undef),
    }
}

fn contains(container: &V, item: &V) -> Result<bool> {
    match (container, item) {
        (V::Str(s), V::Str(i)) => Ok(s.contains(i.as_str())),
        (V::Map(m), V::Str(k)) => Ok(m.iter().any(|(key, _)| key == k)),
        (V::Ns(n), V::Str(k)) => Ok(n.borrow().iter().any(|(key, _)| key == k)),
        (V::List(l), i) => Ok(l.iter().any(|x| eq(x, i))),
        (V::Undef | V::None, _) => Ok(false),
        _ => fail("unsupported `in` operands"),
    }
}

fn binary(op: &str, l: &V, r: &V) -> Result<V> {
    match op {
        "==" => Ok(V::Bool(eq(l, r))),
        "!=" => Ok(V::Bool(!eq(l, r))),
        "<" => Ok(V::Bool(compare(l, r)? == Ordering::Less)),
        ">" => Ok(V::Bool(compare(l, r)? == Ordering::Greater)),
        "<=" => Ok(V::Bool(compare(l, r)? != Ordering::Greater)),
        ">=" => Ok(V::Bool(compare(l, r)? != Ordering::Less)),
        "~" => Ok(V::Str(format!("{}{}", l.to_text(), r.to_text()))),
        "+" => match (l, r) {
            (V::Str(a), V::Str(b)) => Ok(V::Str(format!("{a}{b}"))),
            (V::List(a), V::List(b)) => Ok(V::List(a.iter().chain(b).cloned().collect())),
            (V::Int(a), V::Int(b)) => Ok(V::Int(a + b)),
            _ => match (num(l), num(r)) {
                (Some(a), Some(b)) => Ok(V::Float(a + b)),
                _ => fail(format!("cannot add {l:?} and {r:?}")),
            },
        },
        "-" | "*" | "/" | "//" | "%" => {
            if let (V::Int(a), V::Int(b)) = (l, r) {
                match op {
                    "-" => return Ok(V::Int(a - b)),
                    "*" => return Ok(V::Int(a * b)),
                    "//" | "%" if *b == 0 => return fail("division by zero"),
                    "//" => return Ok(V::Int(a.div_euclid(*b))),
                    "%" => return Ok(V::Int(a.rem_euclid(*b))),
                    _ => {}
                }
            }
            match (num(l), num(r)) {
                (Some(a), Some(b)) => Ok(V::Float(match op {
                    "-" => a - b,
                    "*" => a * b,
                    "/" => a / b,
                    "//" => (a / b).floor(),
                    _ => a - b * (a / b).floor(),
                })),
                _ => fail(format!("cannot apply `{op}` to {l:?} and {r:?}")),
            }
        }
        other => fail(format!("unknown operator `{other}`")),
    }
}

fn filter(name: &str, v: V, args: &[V]) -> Result<V> {
    match name {
        "trim" => Ok(V::Str(v.to_text().trim().to_owned())),
        "safe" => Ok(v),
        "string" => Ok(V::Str(v.to_text())),
        "upper" => Ok(V::Str(v.to_text().to_uppercase())),
        "lower" => Ok(V::Str(v.to_text().to_lowercase())),
        "default" | "d" => {
            let by_truth = args.get(1).is_some_and(V::truthy);
            let missing = matches!(v, V::Undef) || (by_truth && !v.truthy());
            Ok(if missing {
                args.first().cloned().unwrap_or(V::Str(String::new()))
            } else {
                v
            })
        }
        "length" | "count" => match &v {
            V::Str(s) => Ok(V::Int(s.chars().count() as i64)),
            V::List(l) => Ok(V::Int(l.len() as i64)),
            V::Map(m) => Ok(V::Int(m.len() as i64)),
            V::Ns(n) => Ok(V::Int(n.borrow().len() as i64)),
            other => fail(format!("{other:?} has no length")),
        },
        "tojson" => {
            if matches!(v, V::Undef) {
                return fail("cannot serialise an undefined value");
            }
            let mut s = String::new();
            write_json(&v, &mut s);
            Ok(V::Str(s))
        }
        "items" => match v {
            V::Map(m) => Ok(V::List(
                m.into_iter()
                    .map(|(k, x)| V::List(vec![V::Str(k), x]))
                    .collect(),
            )),
            V::Undef | V::None => Ok(V::List(Vec::new())),
            other => fail(format!("{other:?} has no items")),
        },
        "list" => match v {
            V::List(_) => Ok(v),
            V::Map(m) => Ok(V::List(m.into_iter().map(|(k, _)| V::Str(k)).collect())),
            V::Str(s) => Ok(V::List(s.chars().map(|c| V::Str(c.to_string())).collect())),
            other => fail(format!("cannot list {other:?}")),
        },
        "first" => match v {
            V::List(l) => Ok(l.into_iter().next().unwrap_or(V::Undef)),
            other => fail(format!("cannot take first of {other:?}")),
        },
        "last" => match v {
            V::List(l) => Ok(l.into_iter().next_back().unwrap_or(V::Undef)),
            other => fail(format!("cannot take last of {other:?}")),
        },
        "join" => match v {
            V::List(l) => {
                let sep = args.first().map(V::to_text).unwrap_or_default();
                Ok(V::Str(
                    l.iter().map(V::to_text).collect::<Vec<_>>().join(&sep),
                ))
            }
            other => fail(format!("cannot join {other:?}")),
        },
        other => fail(format!("unknown filter `{other}`")),
    }
}

fn test(name: &str, v: &V, args: &[V]) -> Result<bool> {
    Ok(match name {
        "defined" => !matches!(v, V::Undef),
        "undefined" => matches!(v, V::Undef),
        "none" => matches!(v, V::None),
        "string" => matches!(v, V::Str(_)),
        "mapping" => matches!(v, V::Map(_) | V::Ns(_)),
        "iterable" | "sequence" => v.is_iterable(),
        "true" => matches!(v, V::Bool(true)),
        "false" => matches!(v, V::Bool(false)),
        "number" => matches!(v, V::Int(_) | V::Float(_)),
        "integer" => matches!(v, V::Int(_)),
        "boolean" => matches!(v, V::Bool(_)),
        "eq" | "equalto" => args.first().is_some_and(|a| eq(v, a)),
        "ne" => args.first().is_some_and(|a| !eq(v, a)),
        other => return fail(format!("unknown test `{other}`")),
    })
}

fn call_method(target: &V, method: &str, args: &[V]) -> Result<V> {
    match (target, method) {
        (V::Str(s), "startswith" | "endswith") => {
            let candidates: Vec<String> = match args.first() {
                Some(V::List(l)) => l.iter().map(V::to_text).collect(),
                Some(other) => vec![other.to_text()],
                None => return fail(format!("`{method}` needs an argument")),
            };
            Ok(V::Bool(candidates.iter().any(|c| {
                if method == "startswith" {
                    s.starts_with(c.as_str())
                } else {
                    s.ends_with(c.as_str())
                }
            })))
        }
        (V::Str(s), "strip") => Ok(V::Str(s.trim().to_owned())),
        (V::Str(s), "lstrip") => Ok(V::Str(s.trim_start().to_owned())),
        (V::Str(s), "rstrip") => Ok(V::Str(s.trim_end().to_owned())),
        (V::Str(s), "upper") => Ok(V::Str(s.to_uppercase())),
        (V::Str(s), "lower") => Ok(V::Str(s.to_lowercase())),
        (V::Str(s), "replace") => match args {
            [from, to, ..] => Ok(V::Str(s.replace(&from.to_text(), &to.to_text()))),
            _ => fail("`replace` needs two arguments"),
        },
        (V::Str(s), "split") => Ok(V::List(match args.first() {
            Some(sep) => s
                .split(sep.to_text().as_str())
                .map(|p| V::Str(p.to_owned()))
                .collect(),
            None => s.split_whitespace().map(|p| V::Str(p.to_owned())).collect(),
        })),
        (V::Map(m), "items") => filter("items", V::Map(m.clone()), &[]),
        (V::Map(m), "keys") => filter("list", V::Map(m.clone()), &[]),
        (V::Map(m), "values") => Ok(V::List(m.iter().map(|(_, v)| v.clone()).collect())),
        (V::Map(m), "get") => match args.first() {
            Some(V::Str(k)) => Ok(m.iter().find(|(key, _)| key == k).map_or_else(
                || args.get(1).cloned().unwrap_or(V::None),
                |(_, v)| v.clone(),
            )),
            _ => fail("`get` needs a string key"),
        },
        (V::Undef, _) => fail(format!("`{method}` called on an undefined value")),
        _ => fail(format!("unknown method `{method}`")),
    }
}
