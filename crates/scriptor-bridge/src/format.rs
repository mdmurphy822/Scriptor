//! Per-region content cleanup for recognised layout regions: label mapping,
//! content normalisation by label, and the page-level merges of formula
//! numbers, hyphenated text blocks and bullet points.

use std::collections::HashSet;

/// The recognition task a layout region is routed to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionTask {
    Text,
    Table,
    Formula,
    Skip,
}

/// One recognised layout region of a page. `bbox` is in the normalised
/// 0..1000 space of the page and `content` is the recognised text, `None`
/// when no recognition was obtained. `repetition_stopped` is whether the
/// seat reported that its repetition stop ended the recognition.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionResult {
    pub label: String,
    pub task: RegionTask,
    pub bbox: Option<[f64; 4]>,
    pub content: Option<String>,
    pub repetition_stopped: bool,
}

/// The regions of one page in reading order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PageResult {
    pub regions: Vec<RegionResult>,
}

/// A region after formatting: `label` is the mapped label and `native_label`
/// the layout label it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct FormattedRegion {
    pub index: usize,
    pub label: String,
    pub native_label: String,
    pub content: Option<String>,
    pub bbox: Option<[f64; 4]>,
}

const IMAGE_LABELS: [&str; 2] = ["chart", "image"];
const TEXT_LABELS: [&str; 12] = [
    "abstract",
    "algorithm",
    "content",
    "doc_title",
    "figure_title",
    "paragraph_title",
    "reference_content",
    "text",
    "vertical_text",
    "vision_footnote",
    "seal",
    "formula_number",
];
const TABLE_LABELS: [&str; 1] = ["table"];
const FORMULA_LABELS: [&str; 2] = ["display_formula", "inline_formula"];
const REPEAT_MIN_UNIT: usize = 10;
const REPEAT_MIN_COUNT: usize = 10;
const REPEAT_CLEAN_LENGTH: usize = 2048;
const BULLET_LEFT_ALIGN: f64 = 10.0;

/// Whether `c` is whitespace by the rules of Unicode white space plus the
/// four ASCII information separators.
pub(crate) fn is_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

/// Removes leading whitespace.
pub(crate) fn trim_start(text: &str) -> &str {
    text.trim_start_matches(is_space)
}

/// Removes trailing whitespace.
pub(crate) fn trim_end(text: &str) -> &str {
    text.trim_end_matches(is_space)
}

/// Removes leading and trailing whitespace.
pub(crate) fn trim(text: &str) -> &str {
    trim_end(trim_start(text))
}

/// Whether `c` is a decimal digit of the ASCII, Arabic-Indic, Extended
/// Arabic-Indic, Devanagari or fullwidth ranges.
fn is_decimal_digit(c: char) -> bool {
    c.is_ascii_digit()
        || matches!(
            c,
            '\u{660}'..='\u{669}' | '\u{6f0}'..='\u{6f9}' | '\u{966}'..='\u{96f}' | '\u{ff10}'..='\u{ff19}'
        )
}

/// Maps a layout label to the standard label: charts and images become
/// `image`, text-task labels `text`, and table and formula labels `table`
/// and `formula`; any other label is returned unchanged.
pub fn map_label(label: &str) -> String {
    let mapped = if IMAGE_LABELS.contains(&label) {
        "image"
    } else if TEXT_LABELS.contains(&label) {
        "text"
    } else if TABLE_LABELS.contains(&label) {
        "table"
    } else if FORMULA_LABELS.contains(&label) {
        "formula"
    } else {
        label
    };
    mapped.to_string()
}

/// Removes leading literal `\t` sequences and surrounding whitespace.
fn strip_literal_tabs(content: &str) -> String {
    let mut start = content;
    while let Some(rest) = start.strip_prefix("\\t") {
        start = rest;
    }
    let start = trim_start(start);
    let (mut body, newline) = match start.strip_suffix('\n') {
        Some(body) if body.ends_with("\\t") => (body, "\n"),
        _ => (start, ""),
    };
    while let Some(rest) = body.strip_suffix("\\t") {
        body = rest;
    }
    trim_end(&format!("{body}{newline}")).to_string()
}

/// Shortens every run of three or more repetitions of `unit` to three.
fn collapse_runs(text: &str, unit: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(position) = rest.find(unit) {
        out.push_str(&rest[..position]);
        rest = &rest[position..];
        let mut count = 0usize;
        while let Some(next) = rest.strip_prefix(unit) {
            rest = next;
            count += 1;
        }
        for _ in 0..count.min(3) {
            out.push_str(unit);
        }
    }
    out.push_str(rest);
    out
}

/// Truncates `text` at the first place a unit of ten or more characters
/// repeats ten or more times in a row, keeping one copy of the unit.
fn find_consecutive_repeat(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let length = chars.len();
    if length < REPEAT_MIN_UNIT * REPEAT_MIN_COUNT {
        return None;
    }
    let max_unit = length / REPEAT_MIN_COUNT;
    if max_unit < REPEAT_MIN_UNIT {
        return None;
    }
    for start in 0..length {
        for unit in REPEAT_MIN_UNIT..=max_unit {
            if start + unit * REPEAT_MIN_COUNT > length {
                break;
            }
            let span = unit * (REPEAT_MIN_COUNT - 1);
            if (start..start + span).all(|i| chars[i] == chars[i + unit]) {
                return Some(chars[..start + unit].iter().collect());
            }
        }
    }
    None
}

/// Removes runaway repetition from long content: a consecutively repeated
/// unit is truncated to one copy, and a line that makes up at least 80% of
/// the lines of a text of ten or more lines is cut at its first run of three.
fn clean_repeated_content(content: &str) -> String {
    let stripped = trim(content);
    if stripped.is_empty() {
        return content.to_string();
    }
    if stripped.chars().count() > REPEAT_MIN_UNIT * REPEAT_MIN_COUNT
        && let Some(result) = find_consecutive_repeat(stripped)
    {
        return result;
    }
    let lines: Vec<&str> = content
        .split('\n')
        .map(trim)
        .filter(|line| !line.is_empty())
        .collect();
    let total = lines.len();
    if total < REPEAT_MIN_COUNT {
        return content.to_string();
    }
    let mut counts: Vec<(&str, usize)> = Vec::new();
    for line in &lines {
        match counts.iter_mut().find(|entry| entry.0 == *line) {
            Some(entry) => entry.1 += 1,
            None => counts.push((line, 1)),
        }
    }
    let mut common = counts[0];
    for entry in &counts {
        if entry.1 > common.1 {
            common = *entry;
        }
    }
    let (common, count) = common;
    if count < REPEAT_MIN_COUNT || (count as f64) / (total as f64) < 0.8 {
        return content.to_string();
    }
    for (i, line) in lines.iter().enumerate() {
        if *line != common {
            continue;
        }
        let run = lines[i..(i + 3).min(total)]
            .iter()
            .filter(|candidate| **candidate == common)
            .count();
        if run >= 3 {
            let original: Vec<&str> = content.split('\n').collect();
            let mut non_empty = 0usize;
            for (index, line) in original.iter().enumerate() {
                if !trim(line).is_empty() {
                    non_empty += 1;
                    if non_empty == i + 1 {
                        return original[..=index].join("\n");
                    }
                }
            }
            break;
        }
    }
    content.to_string()
}

/// Parses one inline formula `$ ... $` starting at the `$` at `open`: the
/// matched end (exclusive) and the formula span, or `None`.
fn inline_formula_at(chars: &[char], open: usize) -> Option<(usize, usize, usize)> {
    let length = chars.len();
    let after = open + 1;
    let mut spaces = 0usize;
    while after + spaces < length && is_space(chars[after + spaces]) {
        spaces += 1;
    }
    for taken in (0..=spaces).rev() {
        let group_start = after + taken;
        let mut position = group_start;
        loop {
            match chars.get(position) {
                Some('$') | None => break,
                Some('\\') => match chars.get(position + 1) {
                    Some(next) if *next != '\n' => position += 2,
                    _ => break,
                },
                Some(_) => position += 1,
            }
            let mut tail = position;
            while tail < length && is_space(chars[tail]) {
                tail += 1;
            }
            if chars.get(tail) == Some(&'$') && chars.get(tail + 1) != Some(&'$') {
                return Some((tail + 1, group_start, position));
            }
        }
    }
    None
}

/// Normalises inline formula spacing: `$ x $` becomes `$x$`, and a space
/// separates the delimiters from adjacent alphanumeric text.
fn normalize_inline_formula(content: &str) -> String {
    if !content.contains('$') {
        return content.to_string();
    }
    let chars: Vec<char> = content.chars().collect();
    let mut parts = String::new();
    let mut replaced = false;
    let mut last_end = 0usize;
    let mut position = 0usize;
    while position < chars.len() {
        let opens = chars[position] == '$' && (position == 0 || chars[position - 1] != '$');
        let found = if opens {
            inline_formula_at(&chars, position)
        } else {
            None
        };
        let Some((end, group_start, group_end)) = found else {
            position += 1;
            continue;
        };
        let formula: String = chars[group_start..group_end].iter().collect();
        let formula = trim(&formula);
        if !formula.is_empty() {
            let mut before: String = chars[last_end..position].iter().collect();
            if before.chars().last().is_some_and(char::is_alphanumeric) {
                before.push(' ');
            }
            parts.push_str(&before);
            parts.push('$');
            parts.push_str(formula);
            parts.push('$');
            if chars.get(end).is_some_and(|c| c.is_alphanumeric()) {
                parts.push(' ');
            }
            last_end = end;
            replaced = true;
        }
        position = end;
    }
    if !replaced {
        return content.to_string();
    }
    parts.extend(chars[last_end..].iter());
    parts
}

/// Cleans recognised content: literal `\t` edges, long runs of dots, middle
/// dots and underscores, runaway repetition in long content and inline
/// formula spacing.
pub fn clean_content(content: &str) -> String {
    let mut text = strip_literal_tabs(content);
    for unit in [".", "\u{b7}", "_", "\\_"] {
        text = collapse_runs(&text, unit);
    }
    if text.chars().count() >= REPEAT_CLEAN_LENGTH {
        text = clean_repeated_content(&text);
    }
    trim(&normalize_inline_formula(&text)).to_string()
}

/// Whether cleaning `content` removes runaway repetition: after literal tab
/// and leader-run cleaning, a text long enough to be cleaned for repetition
/// that the repetition cleaning shortens.
pub fn repetition_collapsed(content: &str) -> bool {
    let mut text = strip_literal_tabs(content);
    for unit in [".", "\u{b7}", "_", "\\_"] {
        text = collapse_runs(&text, unit);
    }
    text.chars().count() >= REPEAT_CLEAN_LENGTH && clean_repeated_content(&text) != text
}

/// The length in characters of the longest run of one leader character
/// (`.`, a middle dot or `_`) in `content`.
pub fn longest_leader_run(content: &str) -> usize {
    let mut longest = 0usize;
    let mut run = 0usize;
    let mut previous = None;
    for c in content.chars() {
        if matches!(c, '.' | '\u{b7}' | '_') {
            run = if previous == Some(c) { run + 1 } else { 1 };
            longest = longest.max(run);
        } else {
            run = 0;
        }
        previous = Some(c);
    }
    longest
}

/// Splits a leading run of decimal digits, or one ASCII letter, from `text`.
fn take_enumerator_symbol(text: &str) -> Option<(&str, &str)> {
    let digits: usize = text
        .chars()
        .take_while(|c| is_decimal_digit(*c))
        .map(char::len_utf8)
        .sum();
    if digits > 0 {
        return Some(text.split_at(digits));
    }
    let first = text.chars().next()?;
    first.is_ascii_alphabetic().then(|| text.split_at(1))
}

/// Rewrites a leading `(1)` or `（a）` enumerator to `(1) `.
fn parenthesized_enumerator(content: &str) -> Option<String> {
    if content.contains('\n') {
        return None;
    }
    let mut chars = content.chars();
    if !matches!(chars.next()?, '(' | '（') {
        return None;
    }
    let (symbol, after) = take_enumerator_symbol(chars.as_str())?;
    let mut tail = after.chars();
    if !matches!(tail.next()?, ')' | '）') {
        return None;
    }
    Some(format!("({symbol}) {}", trim_start(tail.as_str())))
}

/// Rewrites a leading `1.`, `a)` or `1）` enumerator to carry one space.
fn trailing_enumerator(content: &str) -> Option<String> {
    if content.contains('\n') {
        return None;
    }
    let (symbol, after) = take_enumerator_symbol(content)?;
    let mut tail = after.chars();
    let separator = match tail.next()? {
        '.' => '.',
        ')' | '）' => ')',
        _ => return None,
    };
    Some(format!("{symbol}{separator} {}", trim_start(tail.as_str())))
}

/// Doubles every newline that has no newline beside it.
fn double_single_newlines(content: &str) -> String {
    let chars: Vec<char> = content.chars().collect();
    let mut out = String::with_capacity(content.len() + 8);
    for (i, c) in chars.iter().enumerate() {
        out.push(*c);
        let alone = *c == '\n'
            && (i == 0 || chars[i - 1] != '\n')
            && chars.get(i + 1).is_none_or(|next| *next != '\n');
        if alone {
            out.push('\n');
        }
    }
    out
}

/// Removes a leading run of `#` and the whitespace after it.
fn strip_leading_hashes(content: &str) -> &str {
    let rest = content.trim_start_matches('#');
    if rest.len() == content.len() {
        content
    } else {
        trim_start(rest)
    }
}

/// Formats the content of one region by its mapped `label` and layout
/// `native_label`: tables and formulas keep their markup, titles gain
/// heading markers, formulas are wrapped in `$$` fences, and text gets
/// closed code fences, bullet and enumerator normalisation and doubled
/// newlines.
pub fn format_content(content: &str, label: &str, native_label: &str) -> String {
    let mut content =
        if (label == "table" && content.starts_with("<table") && content.ends_with("</table>"))
            || (label == "formula" && content.starts_with("$$") && content.ends_with("$$"))
        {
            trim(content).to_string()
        } else {
            clean_content(content)
        };
    if native_label == "doc_title" {
        content = format!("# {}", strip_leading_hashes(&content));
    } else if native_label == "paragraph_title" {
        if content.starts_with("- ") || content.starts_with("* ") {
            content = trim_start(&content[2..]).to_string();
        }
        let bare = strip_leading_hashes(&content);
        content = format!("## {}", trim_start(bare));
    }
    if label == "formula" {
        if content.starts_with("$$") || content.starts_with("\\[") || content.starts_with("\\(") {
            content = trim(&content[2..]).to_string();
        }
        if content.ends_with("$$") || content.ends_with("\\]") || content.ends_with("\\)") {
            content = trim(&content[..content.len() - 2]).to_string();
        }
        content = format!("$$\n{content}\n$$");
    }
    if label == "text" {
        if content.starts_with("```") && !content.ends_with("```") {
            content.push_str("\n```");
        }
        if content.starts_with('\u{b7}')
            || content.starts_with('\u{2022}')
            || content
                .chars()
                .next()
                .is_some_and(crate::textutil::is_symbol_font_bullet)
            || content.starts_with("* ")
        {
            let skip = content.chars().next().map_or(0, char::len_utf8);
            content = format!("- {}", trim_start(&content[skip..]));
        }
        if let Some(rewritten) = parenthesized_enumerator(&content) {
            content = rewritten;
        }
        if let Some(rewritten) = trailing_enumerator(&content) {
            content = rewritten;
        }
        content = double_single_newlines(&content);
    }
    content
}

/// Whether a hyphen-joined token looks like a word: three to thirty-one
/// lowercase letters, apostrophes or hyphens starting with a letter and
/// free of doubled hyphens and apostrophes.
fn is_likely_valid_merged_word(word: &str) -> bool {
    let token = trim(word).to_lowercase();
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    let rest = chars.as_str();
    let length = rest.chars().count();
    first.is_ascii_lowercase()
        && (2..=30).contains(&length)
        && rest
            .chars()
            .all(|c| c.is_ascii_lowercase() || c == '\'' || c == '-')
        && !token.contains("--")
        && !token.contains("''")
}

fn content_of(region: &FormattedRegion) -> &str {
    region.content.as_deref().unwrap_or("")
}

/// Joins a text block ending in a hyphen with the next text block that
/// starts lowercase when the joined word is plausible.
fn merge_text_blocks(regions: Vec<FormattedRegion>) -> Vec<FormattedRegion> {
    let mut merged: Vec<FormattedRegion> = Vec::new();
    let mut skipped: HashSet<usize> = HashSet::new();
    for (i, block) in regions.iter().enumerate() {
        if skipped.contains(&i) {
            continue;
        }
        let content = content_of(block);
        let stripped = trim_end(content);
        if block.label != "text" || !stripped.ends_with('-') {
            merged.push(block.clone());
            continue;
        }
        let mut joined = false;
        for (j, next) in regions.iter().enumerate().skip(i + 1) {
            if next.label != "text" {
                continue;
            }
            let next_content = content_of(next);
            let next_stripped = trim_start(next_content);
            if !next_stripped.chars().next().is_some_and(char::is_lowercase) {
                continue;
            }
            let without_hyphen = &stripped[..stripped.len() - 1];
            let before = without_hyphen
                .split(is_space)
                .rfind(|word| !word.is_empty());
            let after = next_stripped.split(is_space).find(|word| !word.is_empty());
            if let (Some(before), Some(after)) = (before, after)
                && is_likely_valid_merged_word(&format!("{before}{after}"))
            {
                let mut replacement = block.clone();
                replacement.content = Some(format!("{without_hyphen}{next_stripped}"));
                merged.push(replacement);
                skipped.insert(j);
                joined = true;
            }
            break;
        }
        if !joined {
            merged.push(block.clone());
        }
    }
    for (index, block) in merged.iter_mut().enumerate() {
        block.index = index;
    }
    merged
}

/// Removes the parentheses around a formula number.
fn clean_formula_number(number: &str) -> String {
    let number = trim(number);
    if (number.starts_with('(') && number.ends_with(')'))
        || (number.starts_with('（') && number.ends_with('）'))
    {
        let mut chars = number.chars();
        chars.next();
        chars.next_back();
        return chars.as_str().to_string();
    }
    number.to_string()
}

/// Appends `\tag{number}` to a `$$` fenced formula.
fn tagged_formula(formula: &FormattedRegion, number: &FormattedRegion) -> FormattedRegion {
    let mut tagged = formula.clone();
    let content = content_of(formula);
    if let Some(body) = content.strip_suffix("\n$$") {
        let number = clean_formula_number(content_of(number));
        tagged.content = Some(format!("{body} \\tag{{{number}}}\n$$"));
    }
    tagged
}

/// Folds each formula number into its adjacent formula as a `\tag`, and
/// drops formula numbers that have no formula after them.
fn merge_formula_numbers(regions: Vec<FormattedRegion>) -> Vec<FormattedRegion> {
    let mut merged: Vec<FormattedRegion> = Vec::new();
    let mut skipped: HashSet<usize> = HashSet::new();
    for (i, block) in regions.iter().enumerate() {
        if skipped.contains(&i) {
            continue;
        }
        let next = regions.get(i + 1);
        if block.native_label == "formula_number" {
            if let Some(formula) = next.filter(|next| next.label == "formula") {
                merged.push(tagged_formula(formula, block));
                skipped.insert(i + 1);
            }
            continue;
        }
        if block.label == "formula"
            && let Some(number) = next.filter(|next| next.native_label == "formula_number")
        {
            merged.push(tagged_formula(block, number));
            skipped.insert(i + 1);
            continue;
        }
        merged.push(block.clone());
    }
    for (index, block) in merged.iter_mut().enumerate() {
        block.index = index;
    }
    merged
}

/// Gives a text block the `- ` marker when it sits between two bulleted
/// text blocks and shares their left edge.
fn format_bullet_points(mut regions: Vec<FormattedRegion>) -> Vec<FormattedRegion> {
    if regions.len() < 3 {
        return regions;
    }
    for i in 1..regions.len() - 1 {
        let all_text = regions[i - 1..=i + 1]
            .iter()
            .all(|block| block.native_label == "text");
        if !all_text || content_of(&regions[i]).starts_with("- ") {
            continue;
        }
        if !(content_of(&regions[i - 1]).starts_with("- ")
            && content_of(&regions[i + 1]).starts_with("- "))
        {
            continue;
        }
        let (Some(current), Some(previous), Some(next)) =
            (regions[i].bbox, regions[i - 1].bbox, regions[i + 1].bbox)
        else {
            continue;
        };
        if (current[0] - previous[0]).abs() <= BULLET_LEFT_ALIGN
            && (current[0] - next[0]).abs() <= BULLET_LEFT_ALIGN
        {
            let content = format!("- {}", content_of(&regions[i]));
            regions[i].content = Some(content);
        }
    }
    regions
}

/// Formats the regions of one page: labels are mapped, content is
/// formatted, regions without content are dropped (images are kept), and
/// formula numbers, hyphenated text blocks and bullet points are merged.
/// Regions keep their given reading order.
pub fn format_page(regions: &[RegionResult]) -> Vec<FormattedRegion> {
    let mut formatted: Vec<FormattedRegion> = Vec::new();
    for region in regions {
        let label = map_label(&region.label);
        let content = region
            .content
            .as_deref()
            .map(|content| format_content(content, &label, &region.label));
        if label != "image" && content.as_deref().is_none_or(|text| trim(text).is_empty()) {
            continue;
        }
        formatted.push(FormattedRegion {
            index: formatted.len(),
            label,
            native_label: region.label.clone(),
            content,
            bbox: region.bbox,
        });
    }
    let formatted = merge_formula_numbers(formatted);
    let formatted = merge_text_blocks(formatted);
    format_bullet_points(formatted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_region(label: &str, content: &str, left: f64) -> RegionResult {
        RegionResult {
            label: label.to_string(),
            task: RegionTask::Text,
            bbox: Some([left, 0.0, left + 100.0, 50.0]),
            content: Some(content.to_string()),
            repetition_stopped: false,
        }
    }

    fn contents(regions: &[FormattedRegion]) -> Vec<String> {
        regions.iter().map(|r| content_of(r).to_string()).collect()
    }

    #[test]
    fn labels_map_to_standard_labels() {
        assert_eq!(map_label("chart"), "image");
        assert_eq!(map_label("doc_title"), "text");
        assert_eq!(map_label("inline_formula"), "formula");
        assert_eq!(map_label("footnote"), "footnote");
    }

    #[test]
    fn cleaning_collapses_runs_and_literal_tabs() {
        assert_eq!(clean_content("\\t\\t  a.....b\\t"), "a...b");
        assert_eq!(clean_content("x_______y"), "x___y");
        assert_eq!(clean_content("x\\_\\_\\_\\_\\_y"), "x\\_\\_\\_y");
        assert_eq!(clean_content("keep ..  "), "keep ..");
    }

    #[test]
    fn repetition_collapse_is_reported_only_when_cleaning_shortens_the_text() {
        let unit = "0123456789ab ";
        assert!(repetition_collapsed(&format!("start {}", unit.repeat(200))));
        assert!(!repetition_collapsed("A short ordinary paragraph."));
        assert!(!repetition_collapsed(&format!(
            "Chapter{}12",
            ".".repeat(5000)
        )));
    }

    #[test]
    fn the_longest_leader_run_counts_one_character_at_a_time() {
        assert_eq!(longest_leader_run("a....b..c"), 4);
        assert_eq!(longest_leader_run("a.._.b"), 2);
        assert_eq!(longest_leader_run("none"), 0);
    }

    #[test]
    fn cleaning_trims_long_consecutive_repeats() {
        let unit = "abcdefghij ";
        let content = format!("start {}", unit.repeat(200));
        assert_eq!(
            clean_content(&content),
            format!("start {}", unit.trim_end())
        );
    }

    #[test]
    fn inline_formulas_are_tightened() {
        assert_eq!(normalize_inline_formula("a $ x + 1 $ b"), "a $x + 1$ b");
        assert_eq!(normalize_inline_formula("see$x$now"), "see $x$ now");
        assert_eq!(normalize_inline_formula("price $5"), "price $5");
        assert_eq!(normalize_inline_formula("$$ x $$"), "$$ x $$");
    }

    #[test]
    fn titles_gain_heading_markers() {
        assert_eq!(format_content("## Intro", "text", "doc_title"), "# Intro");
        assert_eq!(
            format_content("- 1 Setup", "text", "paragraph_title"),
            "## 1 Setup"
        );
    }

    #[test]
    fn formulas_are_fenced() {
        assert_eq!(
            format_content("\\[ E=mc^2 \\]", "formula", "display_formula"),
            "$$\nE=mc^2\n$$"
        );
        assert_eq!(
            format_content("$$a$$", "formula", "display_formula"),
            "$$\na\n$$"
        );
    }

    #[test]
    fn tables_keep_their_markup() {
        let table = "<table><tr><td>1</td></tr></table>";
        assert_eq!(format_content(table, "table", "table"), table);
    }

    #[test]
    fn text_rules_normalise_lists_fences_and_newlines() {
        assert_eq!(format_content("• one", "text", "text"), "- one");
        assert_eq!(format_content("(1)  first", "text", "text"), "(1) first");
        assert_eq!(format_content("（a）first", "text", "text"), "(a) first");
        assert_eq!(format_content("2.next", "text", "text"), "2. next");
        assert_eq!(format_content("b）next", "text", "text"), "b) next");
        assert_eq!(
            format_content("```sh\nls", "text", "text"),
            "```sh\n\nls\n\n```"
        );
        assert_eq!(format_content("a\nb\n\nc", "text", "text"), "a\n\nb\n\nc");
        assert_eq!(format_content("Word", "text", "text"), "Word");
        assert_eq!(format_content("a\nb", "footnote", "footnote"), "a\nb");
    }

    #[test]
    fn blank_regions_are_dropped_and_images_kept() {
        let regions = vec![
            text_region("text", "  ", 0.0),
            RegionResult {
                label: "image".into(),
                task: RegionTask::Skip,
                bbox: Some([0.0, 0.0, 10.0, 10.0]),
                content: None,
                repetition_stopped: false,
            },
            text_region("text", "Body", 0.0),
        ];
        let formatted = format_page(&regions);
        assert_eq!(formatted.len(), 2);
        assert_eq!(formatted[0].label, "image");
        assert_eq!(formatted[1].index, 1);
    }

    #[test]
    fn formula_numbers_become_tags() {
        let regions = vec![
            RegionResult {
                label: "display_formula".into(),
                task: RegionTask::Formula,
                bbox: None,
                content: Some("E=mc^2".into()),
                repetition_stopped: false,
            },
            text_region("formula_number", "(1)", 0.0),
            text_region("formula_number", "(2)", 0.0),
        ];
        let formatted = format_page(&regions);
        assert_eq!(contents(&formatted), vec!["$$\nE=mc^2 \\tag{1}\n$$"]);
    }

    #[test]
    fn hyphenated_text_blocks_merge_when_the_word_is_plausible() {
        let regions = vec![
            text_region("text", "an inter-", 0.0),
            text_region("text", "national treaty", 0.0),
            text_region("text", "Next block", 0.0),
        ];
        let formatted = format_page(&regions);
        assert_eq!(
            contents(&formatted),
            vec!["an international treaty", "Next block"]
        );
        assert_eq!(formatted[1].index, 1);
    }

    #[test]
    fn unplausible_hyphen_merges_are_left_alone() {
        let regions = vec![
            text_region("text", "pre 12-", 0.0),
            text_region("text", "ab", 0.0),
        ];
        assert_eq!(contents(&format_page(&regions)), vec!["pre 12-", "ab"]);
    }

    #[test]
    fn unbulleted_text_between_bullets_gains_the_marker() {
        let regions = vec![
            text_region("text", "- one", 100.0),
            text_region("text", "two", 105.0),
            text_region("text", "- three", 100.0),
        ];
        assert_eq!(
            contents(&format_page(&regions)),
            vec!["- one", "- two", "- three"]
        );
        let offset = vec![
            text_region("text", "- one", 100.0),
            text_region("text", "two", 300.0),
            text_region("text", "- three", 100.0),
        ];
        assert_eq!(contents(&format_page(&offset))[1], "two");
    }
}
