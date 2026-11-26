//! Small text helpers: external link targets, symbol-font bullets, and the
//! split of leading prose rows from an HTML table.

/// The first and last code points of the private-use block that Symbol and
/// Wingdings text layers write bullet glyphs into (U+F0B7, U+F0A7, U+F076,
/// U+F0D8, U+F0FC and the like).
const SYMBOL_FONT_BULLETS: std::ops::RangeInclusive<char> = '\u{F020}'..='\u{F0FF}';

/// Whether `c` is a bullet glyph from a symbol font, a private-use code point
/// that never reads as text.
pub fn is_symbol_font_bullet(c: char) -> bool {
    SYMBOL_FONT_BULLETS.contains(&c)
}

/// Whether `target` is an external link target that is kept: an `http` or
/// `https` URL with a host, or a `mailto` address, with no whitespace or
/// control characters. Every other scheme is rejected.
pub fn is_external_link_target(target: &str) -> bool {
    if target.is_empty() || target.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return false;
    }
    let Some((scheme, rest)) = target.split_once(':') else {
        return false;
    };
    match scheme.to_ascii_lowercase().as_str() {
        "http" | "https" => rest.strip_prefix("//").is_some_and(|host| {
            host.chars()
                .next()
                .is_some_and(|c| !matches!(c, '/' | '?' | '#'))
        }),
        "mailto" => !rest.is_empty(),
        _ => false,
    }
}

/// The result of [`detach_lead_rows`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DetachedLeadRows {
    /// The text of each leading row that was one cell spanning the whole table
    /// width, in table order.
    pub paragraphs: Vec<String>,
    /// The table without those rows and without rows whose cells are all
    /// empty; `None` when no row remains.
    pub table: Option<String>,
}

/// Splits the HTML table in `text` into the leading rows that are one
/// non-empty cell spanning the full table width (prose, not table data) and
/// the rest, and drops every row whose cells are all empty. A table is at least
/// two columns wide for a row to count as spanning. `None` when `text` is not
/// an HTML table or nothing is detached or dropped.
pub fn detach_lead_rows(text: &str) -> Option<DetachedLeadRows> {
    if !text.to_ascii_lowercase().contains("<table") {
        return None;
    }
    let spans = row_spans(text);
    let rows: Vec<Vec<String>> = spans
        .iter()
        .map(|&(start, end)| {
            html_table_rows(&format!("<table>{}</table>", &text[start..end])).unwrap_or_default()
        })
        .map(|mut parsed| parsed.drain(..).next().unwrap_or_default())
        .collect();
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    let blank = |cell: &String| cell.trim().is_empty();
    let mut paragraphs = Vec::new();
    let mut removed = Vec::new();
    let mut leading = true;
    for (index, row) in rows.iter().enumerate() {
        if row.iter().all(blank) {
            removed.push(index);
        } else if leading
            && width >= 2
            && row.len() == width
            && row[1..].iter().all(blank)
            && single_cell(&text[spans[index].0..spans[index].1])
        {
            paragraphs.push(row[0].clone());
            removed.push(index);
        } else {
            leading = false;
        }
    }
    if removed.is_empty() {
        return None;
    }
    let remaining = rows.len() - removed.len();
    if remaining == 0 && paragraphs.is_empty() {
        return None;
    }
    let table = (remaining > 0).then(|| {
        let mut kept = text.to_string();
        for &index in removed.iter().rev() {
            let (start, end) = spans[index];
            kept.replace_range(start..end, "");
        }
        kept
    });
    Some(DetachedLeadRows { paragraphs, table })
}

/// Whether the HTML table row `row` holds exactly one `<td>` or `<th>` cell.
fn single_cell(row: &str) -> bool {
    let lower = row.to_ascii_lowercase();
    let opens = |tag: &str| {
        lower
            .match_indices(tag)
            .filter(|&(at, _)| {
                matches!(
                    lower.as_bytes().get(at + tag.len()),
                    Some(b'>' | b' ' | b'\t' | b'\n' | b'\r')
                )
            })
            .count()
    };
    opens("<td") + opens("<th") == 1
}

/// The `[start, end)` byte ranges of each `<tr>...</tr>` in `html`.
fn row_spans(html: &str) -> Vec<(usize, usize)> {
    let lower = html.to_ascii_lowercase();
    let mut spans = Vec::new();
    let mut at = 0;
    while let Some(offset) = lower[at..].find("<tr") {
        let start = at + offset;
        let after = lower.as_bytes().get(start + 3).copied();
        if !matches!(after, Some(b'>' | b' ' | b'\t' | b'\n' | b'\r' | b'/')) {
            at = start + 3;
            continue;
        }
        let end = lower[start..]
            .find("</tr>")
            .map_or(lower.len(), |close| start + close + "</tr>".len());
        spans.push((start, end));
        at = end;
    }
    spans
}

/// The rows of an HTML `<table>` in `text`, each cell's visible text in
/// order and a `colspan` cell followed by empty cells; `None` when `text`
/// holds no table rows.
fn html_table_rows(text: &str) -> Option<Vec<Vec<String>>> {
    let lower = text.to_ascii_lowercase();
    if !lower.contains("<table") {
        return None;
    }
    let mut rows = Vec::new();
    let mut rest = 0;
    while let Some(start) = find_tag(&lower, "<tr", rest) {
        let end = lower[start..]
            .find("</tr")
            .map_or(lower.len(), |offset| start + offset);
        let mut cells = Vec::new();
        let mut at = start;
        while let Some(open) = next_cell(&lower, at, end) {
            let tag_end = lower[open..end].find('>')? + open;
            let colspan = attribute_number(&lower[open..tag_end], "colspan").unwrap_or(1);
            let close = lower[tag_end..end]
                .find("</t")
                .map_or(end, |offset| tag_end + offset);
            cells.push(cell_text(&text[tag_end + 1..close]));
            cells.extend(std::iter::repeat_n(
                String::new(),
                colspan.saturating_sub(1),
            ));
            at = close.max(tag_end + 1);
        }
        if !cells.is_empty() {
            rows.push(cells);
        }
        rest = end.max(start + 3);
    }
    (!rows.is_empty()).then_some(rows)
}

/// The offset of the next `tag` (not a longer tag name) at or after `from`.
fn find_tag(lower: &str, tag: &str, from: usize) -> Option<usize> {
    let mut at = from;
    while let Some(offset) = lower.get(at..)?.find(tag) {
        let start = at + offset;
        let after = lower[start + tag.len()..].chars().next();
        if matches!(after, Some('>' | ' ' | '\t' | '\n' | '\r' | '/')) {
            return Some(start);
        }
        at = start + tag.len();
    }
    None
}

/// The offset of the next `<td` or `<th` cell tag in `lower[from..end]`.
fn next_cell(lower: &str, from: usize, end: usize) -> Option<usize> {
    let window = &lower[..end];
    let td = find_tag(window, "<td", from);
    let th = find_tag(window, "<th", from);
    match (td, th) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

fn attribute_number(tag: &str, name: &str) -> Option<usize> {
    let after = tag.split(name).nth(1)?.trim_start().strip_prefix('=')?;
    let digits: String = after
        .trim_start()
        .trim_start_matches(['"', '\''])
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits
        .parse()
        .ok()
        .filter(|count| (1..=1000).contains(count))
}

/// An HTML cell's visible text: tags removed (`<br>` as a space), entities
/// decoded, whitespace collapsed.
fn cell_text(html: &str) -> String {
    let mut text = String::new();
    let mut rest = html;
    while let Some(open) = rest.find('<') {
        text.push_str(&rest[..open]);
        match rest[open..].find('>') {
            Some(close) => {
                let tag = rest[open + 1..open + close].trim().to_ascii_lowercase();
                if tag.starts_with("br") || tag.starts_with("/p") || tag.starts_with("p") {
                    text.push(' ');
                }
                rest = &rest[open + close + 1..];
            }
            None => {
                text.push_str(&rest[open..]);
                rest = "";
            }
        }
    }
    text.push_str(rest);
    decode_entities(&text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// `text` with the named entities `&amp;`, `&lt;`, `&gt;`, `&quot;`,
/// `&apos;`, `&nbsp;` and numeric character references decoded; anything
/// else is left as written.
fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let decoded = rest.find(';').filter(|&end| end <= 10).and_then(|end| {
            let name = &rest[1..end];
            let c = match name {
                "amp" => '&',
                "lt" => '<',
                "gt" => '>',
                "quot" => '"',
                "apos" => '\'',
                "nbsp" => ' ',
                _ => {
                    let number = name.strip_prefix('#')?;
                    let code = match number.strip_prefix(['x', 'X']) {
                        Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                        None => number.parse().ok()?,
                    };
                    char::from_u32(code)?
                }
            };
            Some((c, end + 1))
        });
        match decoded {
            Some((c, len)) => {
                out.push(c);
                rest = &rest[len..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_link_targets_are_http_https_and_mailto_only() {
        assert!(is_external_link_target("https://example.org/a"));
        assert!(is_external_link_target("HTTP://example.org"));
        assert!(is_external_link_target("mailto:a@example.org"));
        assert!(!is_external_link_target("javascript:alert(1)"));
        assert!(!is_external_link_target("https:///path"));
        assert!(!is_external_link_target("https://a b"));
        assert!(!is_external_link_target(""));
    }

    #[test]
    fn symbol_font_bullets_are_the_private_use_block() {
        assert!(is_symbol_font_bullet('\u{F0B7}'));
        assert!(!is_symbol_font_bullet('\u{2022}'));
    }

    #[test]
    fn leading_single_cell_rows_detach_as_paragraphs() {
        let table = "<table><tr><td colspan=\"2\">Intro</td></tr>\
                     <tr><td>a</td><td>b</td></tr></table>";
        let detached = detach_lead_rows(table).unwrap();
        assert_eq!(detached.paragraphs, vec!["Intro".to_string()]);
        let kept = detached.table.unwrap();
        assert!(kept.contains("<td>a</td>"));
        assert!(!kept.contains("Intro"));
    }

    #[test]
    fn a_plain_table_detaches_nothing() {
        assert_eq!(
            detach_lead_rows("<table><tr><td>a</td><td>b</td></tr></table>"),
            None
        );
        assert_eq!(detach_lead_rows("not a table"), None);
    }

    #[test]
    fn entities_decode_in_cell_text() {
        assert_eq!(cell_text("a&amp;b&#65;<br>c"), "a&bA c");
    }
}
