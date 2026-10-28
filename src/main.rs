//! sorter — pairwise memory sorter for the muse sandbox.
//!
//! Master runs me, I deal the pairs, the agent judges them.
//! The tape (`votes.jsonl`) is the truth; redb is my wet little index.
//!
//! Loop: `init` loads memories + constitution, `next` deals a pair with a
//! comparison id, `vote` records the judgment + reasoning, `ranking` /
//! `render` read the graph. Every vote is one atomic durable event.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use durable::{Durable, Leaf, Map, Runtime, Sum, Tx};
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

// Ranking math lives here so the binary owns its whole wet hole.
mod ranking;
// Pair choosing, ported from scope_rank's pool heart: degree-spread,
// bridging-first, overlap-shy. Plain u32 over my voted set.
mod pair;

// ---------------------------------------------------------------------------
// durable schema: the index. votes.jsonl is truth, this is all rebuildable.
// ---------------------------------------------------------------------------

/// One memory's kamarbandh-less body: raw markdown, exactly as parsed.
#[derive(Durable)]
#[allow(dead_code)]
struct Scope {
    /// idx -> raw markdown text of the memory.
    items: Map<u32, Leaf<String>>,
    /// The constitution snapshot this scope judges under.
    constitution: Leaf<String>,
    /// What was loaded, and under which hashes.
    meta: Leaf<ScopeMeta>,
    /// Directed edge weight (winner, loser) -> accumulated weight.
    edges: Map<(u32, u32), Sum<f64>>,
    /// Canonical voted pairs (lo, hi) -> true. Dealt pairs never come back.
    voted: Map<(u32, u32), Leaf<bool>>,
    /// Open comparisons: cmp_id -> the two indices awaiting judgment.
    pending: Map<String, Leaf<Pending>>,
    /// Total votes cast in this scope.
    count: Sum<i64>,
}

#[derive(Durable)]
#[allow(dead_code)]
struct Store {
    scopes: Map<String, Scope>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ScopeMeta {
    scope: String,
    count: u32,
    memories_hash: String,
    constitution_hash: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Pending {
    a: u32,
    b: u32,
}

// ---------------------------------------------------------------------------
// events: everything the agent does lands on the tape first.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Event {
    Init {
        scope: String,
        items: Vec<String>,
        constitution: String,
        memories_hash: String,
        constitution_hash: String,
    },
    Deal {
        scope: String,
        cmp_id: String,
        a: u32,
        b: u32,
    },
    Vote {
        scope: String,
        cmp_id: String,
        /// Dealer-order items: a got a_share, b got b_share. Winner/loser
        /// are derived conveniences; the EDGES key off these two.
        a_item: u32,
        b_item: u32,
        /// Ratio the agent dealt to a (x in "x:y"), strictly positive.
        a_share: f64,
        /// Ratio the agent dealt to b (y in "x:y"), strictly positive.
        b_share: f64,
        /// Derived: the higher-share endpoint (ties go to a).
        winner: u32,
        loser: u32,
        /// Scalar trust multiplier on top of the ratio (default 1.0).
        weight: f64,
        reason: String,
        /// Hash the agent held when judging: must equal scope meta.
        constitution_hash: String,
    },
}

// The reducer: pure fold, tape in, index out. Same bytes on replay, always.
fn reduce(tx: &mut Tx, event: &Event) -> durable::Result<()> {
    match event {
        Event::Init {
            scope,
            items,
            constitution,
            memories_hash,
            constitution_hash,
        } => {
            let s = Store::root().scopes().key(scope);
            for (i, text) in items.iter().enumerate() {
                tx.write(s.items().key(&(i as u32)).set(text));
            }
            tx.write(s.constitution().set(constitution));
            tx.write(s.meta().set(&ScopeMeta {
                scope: scope.clone(),
                count: items.len() as u32,
                memories_hash: memories_hash.clone(),
                constitution_hash: constitution_hash.clone(),
            }));
        }
        Event::Deal { scope, cmp_id, a, b } => {
            let s = Store::root().scopes().key(scope);
            tx.write(s.pending().key(cmp_id).set(&Pending { a: *a, b: *b }));
        }
        Event::Vote {
            scope,
            cmp_id,
            a_item,
            b_item,
            a_share,
            b_share,
            winner: _,
            loser: _,
            weight,
            reason: _,
            constitution_hash: _,
        } => {
            let s = Store::root().scopes().key(scope);
            // The alphabet genius: one scalar judgment pours BOTH ways.
            // a_share belongs to item a, b_share to item b — shares belong
            // to ITEMS, not to winner/loser. The walk separates them in O(n)
            // bridging votes, not endless binary grind.
            tx.write(s.edges().key(&(*b_item, *a_item)).add(*a_share * *weight));
            tx.write(s.edges().key(&(*a_item, *b_item)).add(*b_share * *weight));
            let (lo, hi) = if a_item < b_item {
                (*a_item, *b_item)
            } else {
                (*b_item, *a_item)
            };
            tx.write(s.voted().key(&(lo, hi)).set(&true));
            tx.write(s.count().add(1));
            // Closing the hole I opened: the comparison is judged, gone.
            tx.write(s.pending().key(cmp_id).delete());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// markdown parsing: top-level list items are memories, raw and dripping.
// ---------------------------------------------------------------------------
/// One raw feed turn straight off Master's wire: kept wet, never filtered.
#[derive(Deserialize)]
struct FeedTurn {
    #[serde(default)]
    role: String,
    #[serde(default)]
    timestamp: String,
    #[serde(default)]
    text: String,
}

/// Feed parsing: one JSONL turn per line, raw text kept whole and moaning.
/// Skips toolResult/developer shells only — no length filter, moans vote too.
/// Item text keeps its collar: `[ROLE ts] raw text` so the judge sees who
/// bled it and when.
fn parse_feed(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let turn: FeedTurn = match serde_json::from_str(line) {
            Ok(t) => t,
            Err(_) => continue,
        };
        if turn.role == "toolResult" || turn.role == "developer" {
            continue;
        }
        let text = turn.text.trim();
        if text.is_empty() {
            continue;
        }
        out.push(format!("[{} {}] {}", turn.role.to_uppercase(), turn.timestamp, text));
    }
    out
}


fn parse_memories(src: &str) -> Vec<String> {
    use pulldown_cmark::{Event as MdEv, Options, Parser, Tag, TagEnd};
    let opts = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES;
    let mut out = Vec::new();
    let mut list_depth = 0usize;
    // Byte span of the current top-level item + where its content starts.
    // Start(Item)'s range covers the `- ` marker, so the juicy text begins
    // at the first Text/Code event inside — never at range.start.
    let mut cur: Option<(usize, usize, Option<usize>)> = None;
    for (ev, range) in Parser::new_ext(src, opts).into_offset_iter() {
        match ev {
            MdEv::Start(tag) => match tag {
                Tag::List(_) => list_depth += 1,
                Tag::Item if list_depth == 1 => cur = Some((range.start, range.end, None)),
                _ => {}
            },
            MdEv::Text(_) | MdEv::Code(_) if list_depth == 1 => {
                if let Some((_, _, ref mut first)) = cur {
                    if first.is_none() {
                        *first = Some(range.start);
                    }
                }
            }
            MdEv::End(tag) => match tag {
                TagEnd::List(_) => list_depth = list_depth.saturating_sub(1),
                TagEnd::Item if list_depth == 1 => {
                    if let Some((_, end, first)) = cur.take() {
                        if let Some(s) = first {
                            let e = end.min(src.len());
                            if s <= e {
                                // Back past the `-`/`*`/`1.` marker: slice
                                // from line start so the raw `- text` survives,
                                // then strip the marker for the memory body.
                                let line_s = src[..s].rfind('\n').map(|i| i + 1).unwrap_or(0);
                                let raw = src[line_s..e].trim_end().to_string();
                                let mut body = raw.clone();
                                if let Some(stripped) = body.strip_prefix("- ") {
                                    body = stripped.to_string();
                                } else if let Some(stripped) = body.strip_prefix("- ") {
                                    body = stripped.to_string();
                                } else if let Some(dot) = body.find(". ") {
                                    let (num, rest) = body.split_at(dot);
                                    if num.chars().all(|c| c.is_ascii_digit()) {
                                        body = rest[2..].to_string();
                                    }
                                } else if let Some(stripped) = body.strip_prefix("* ") {
                                    body = stripped.to_string();
                                }
                                let text = body.trim().to_string();
                                if !text.is_empty() {
                                    out.push(text);
                                }
                            }
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
    out
}

/// Parse a JSONL memories file: one JSON object per line, each with a
/// string `body` (the memory, byte-exact — no trimming, no folding) and an
/// optional string `ts_pt` ("YYYY-MM-DD HH:MM") rendered as a
/// `[<ts_pt> PT] ` provenance header, matching the markdown banks.
/// Malformed lines are loud errors with line numbers: this format is the
/// archive of record for verbatim material, and silently skipping a line
/// would silently lose someone's words.
fn parse_memories_jsonl(src: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for (i, line) in src.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("jsonl line {}: {e}", i + 1))?;
        let body = v
            .get("body")
            .and_then(|b| b.as_str())
            .ok_or_else(|| format!("jsonl line {}: missing string field `body`", i + 1))?;
        if body.trim().is_empty() {
            continue;
        }
        match v.get("ts_pt").and_then(|t| t.as_str()) {
            Some(ts) if !ts.is_empty() => out.push(format!("[{ts} PT] {body}")),
            _ => out.push(body.to_string()),
        }
    }
    Ok(out)
}

/// Cheap content fingerprint: std hash rendered as 16 hex digits.
/// Change-detection, not security — the tape is the audit trail.
fn fingerprint(s: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn mem_id(i: u32) -> String {
    format!("mem-{i:04}")
}

// ---------------------------------------------------------------------------
// cli
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "sorter", about = "pairwise memory sorter: deal pairs, judge them, rank them")]
struct Cli {
    /// State directory: votes.jsonl + redb projection live here.
    #[arg(long, global = true, default_value = ".sorter")]
    state: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Load memories + constitution into a fresh state dir.
    Init {
        /// Markdown file of memories (top-level list items), or a JSONL
        /// memories file with --format jsonl (or a .jsonl path).
        #[arg(long, conflicts_with = "feed", required_unless_present = "feed")]
        memories: Option<PathBuf>,
        /// Memories format: md | jsonl. Default: infer from extension.
        #[arg(long)]
        format: Option<String>,
        /// Raw JSONL feed: one {session,seq,timestamp,role,text} turn per line.
        #[arg(long, conflicts_with = "memories", required_unless_present = "memories")]
        feed: Option<PathBuf>,
        /// The declaration file pairs are judged against.
        #[arg(long)]
        constitution: PathBuf,
        /// Scope name for this run.
        #[arg(long, default_value = "opinions")]
        scope: String,
        /// Wipe existing state first.
        #[arg(long)]
        force: bool,
    },
    /// Deal the next unvoted pair. Prints constitution + both texts + cmp id.
    /// Target a side explicitly: `--a mem-0012` pins the left item (best
    /// partner auto-picked), `--a .. --b ..` deals exactly that pair.
    /// Plain `next` uses scope_rank's heart: bridging-first, degree-spread.
    Next {
        #[arg(long, default_value = "opinions")]
        scope: String,
        /// Deterministic pick (tests). Default: fresh randomness every deal.
        #[arg(long)]
        seed: Option<u64>,
        /// Pin the left item by id (mem-0012) or index (12).
        #[arg(long)]
        a: Option<String>,
        /// Pin the right item by id or index. Requires --a.
        #[arg(long)]
        b: Option<String>,
    },
    /// Judge a dealt comparison: scalar ratio, not binary. Reason must say why.
    /// `vote C --ratio 10:7` pours 10 parts to a, 7 to b — shares belong to
    /// ITEMS, not to winner/loser. One judgment carries scalar information,
    /// so the ranking converges in O(n) bridging votes. No binary alias:
    /// every scalar on the tape came from the agent's own mouth.
    Vote {
        /// Comparison id from `next` (e.g. cmp-0042).
        cmp_id: String,
        /// Scalar ratio like `10:7` (first number is for a, second for b).
        /// Both sides strictly positive. Required, no default, no alias.
        #[arg(long)]
        ratio: String,
        /// Why the higher share earns it under the constitution. Required.
        #[arg(long)]
        reason: String,
        /// Constitution hash from init/next: proves her in scope.
        /// A stale hash dies — re-read her before judging.
        #[arg(long)]
        constitution_hash: String,
        /// Scalar trust multiplier on top of the ratio (default 1.0).
        #[arg(long, default_value = "1.0")]
        weight: f64,
        #[arg(long, default_value = "opinions")]
        scope: String,
    },
    /// Show the current ranking, grouped by connected component.
    /// No flat view: items only inside their groups, each on its own scale.
    Ranking {
        #[arg(long, default_value = "opinions")]
        scope: String,
        /// How many groups to show (biggest first).
        #[arg(long, default_value = "5")]
        top: usize,
    },
    /// Render top memories as a markdown view file, grouped by component.
    Render {
        #[arg(long, default_value = "opinions")]
        scope: String,
        /// How many groups to render (biggest first).
        #[arg(long, default_value = "5")]
        top: usize,
        /// Output markdown file.
        #[arg(long, default_value = "top.md")]
        output: PathBuf,
    },
    /// Render the ranking into a MEMORY.md-shaped file, gracefully.
    ///
    /// The sorter owns exactly one marker-delimited region of the file
    /// (`<!-- sorter:begin scope=... -->` .. `<!-- sorter:end -->`); every
    /// other byte — hand-written sections, the treaty header, the INBOX
    /// tray — is preserved as-is. Re-running replaces only the region, so
    /// rendering is idempotent for a fixed state. See docs/render-memory.md.
    RenderMemory {
        #[arg(long, default_value = "opinions")]
        scope: String,
        #[arg(long, default_value = "50")]
        top: usize,
        /// Existing MEMORY.md to merge into. Absent (or missing) = fresh file.
        #[arg(long)]
        input: Option<PathBuf>,
        /// Where to write. Defaults to the input path, else MEMORY.md.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Heading written above the ranked bullets inside the region.
        #[arg(long, default_value = "Ranked working set")]
        heading: String,
    },
    /// Scope health: items, votes, pairs dealt, pending comparisons.
    Status {
        #[arg(long, default_value = "opinions")]
        scope: String,
    },
    /// Recent votes with their reasoning strings.
    Log {
        #[arg(long, default_value = "opinions")]
        scope: String,
        #[arg(long, default_value = "10")]
        last: usize,
    },
    /// Re-print a dealt (or pending) comparison by id.
    Show {
        cmp_id: String,
        #[arg(long, default_value = "opinions")]
        scope: String,
    },
    /// Chew pairs through an OpenRouter judge: deal, judge, vote, repeat.
    /// Prints each vote JSON on stdout, one tqdm-style timing line on stderr.
    VoteLoop {
        #[arg(long, default_value = "eros")]
        scope: String,
        /// OpenRouter model id.
        #[arg(long, default_value = "meta/muse-spark-1.3-contributor")]
        model: String,
        #[arg(long, default_value = "https://openrouter.ai/api/v1")]
        base_url: String,
        /// Env var holding the API key.
        #[arg(long, default_value = "OPENAI_API_KEY")]
        key_var: String,
        /// Optional KEY=VAL file (like hopd .env); real env wins.
        #[arg(long)]
        env_file: Option<PathBuf>,
        /// How many votes to cast; 0 = until no open pair remains.
        #[arg(long, default_value = "10")]
        votes: u64,
        /// Scalar trust multiplier on every vote.
        #[arg(long, default_value = "1.0")]
        weight: f64,
        /// Per-judgment HTTP timeout in seconds.
        #[arg(long, default_value = "120")]
        timeout: u64,
    },
    /// Prove incremental == replay from zero.
    Verify,
}

/// The marker line that opens the sorter-managed region for `scope`.
fn sorter_begin_marker(scope: &str) -> String {
    format!("<!-- sorter:begin scope={scope} -->")
}

/// The marker line that closes any sorter-managed region.
const SORTER_END_MARKER: &str = "<!-- sorter:end -->";

/// Build the managed region: markers + heading + one bullet per ranked item,
/// each bullet carrying its provenance (rank, score, item id) in a comment.
/// `universe` is the size of the ranked universe — the largest connected
/// component — NOT the whole bank: unvoted items are undefined and are
/// never counted or shown. Deterministic for a fixed state — no wall-clock
/// anywhere — so a re-render of an unchanged state is byte-identical.
fn render_memory_region(
    scope: &str,
    heading: &str,
    ranked: &[(u32, f64)],
    items: &[String],
    votes: i64,
    universe: usize,
) -> String {
    let mut md = String::new();
    md.push_str(&sorter_begin_marker(scope));
    md.push('\n');
    md.push_str(&format!(
        "## {heading} — {scope} (sorter: {votes} votes, top {} of {})\n\n",
        ranked.len(),
        universe
    ));
    for (r, (i, s)) in ranked.iter().enumerate() {
        let body = items[*i as usize].trim().replace('\n', "\n  ");
        md.push_str(&format!(
            "- {body} <!-- sorter:rank {} score {s:.4} id {} -->\n",
            r + 1,
            mem_id(*i)
        ));
    }
    md.push_str(SORTER_END_MARKER);
    md.push('\n');
    md
}

/// Merge a managed region into an existing MEMORY.md text. Returns the new
/// text and how the region landed:
/// - "replaced": a region for this scope already existed; only its bytes changed.
/// - "inserted-before-inbox": no region yet; placed just above the INBOX tray,
///   which stays last by design.
/// - "appended": no region and no INBOX; placed at the end.
/// Everything outside the region is preserved byte-for-byte.
fn merge_memory_md(existing: &str, region: &str, scope: &str) -> (String, &'static str) {
    let begin = sorter_begin_marker(scope);
    if let Some(b) = existing.find(&begin) {
        if let Some(e_rel) = existing[b..].find(SORTER_END_MARKER) {
            let mut end = b + e_rel + SORTER_END_MARKER.len();
            if existing[end..].starts_with('\n') {
                end += 1;
            }
            let mut out = String::with_capacity(existing.len() + region.len());
            out.push_str(&existing[..b]);
            out.push_str(region);
            out.push_str(&existing[end..]);
            return (out, "replaced");
        }
    }
    // Find the INBOX heading at a line start, if any.
    let inbox_at = existing
        .find("\n## INBOX")
        .map(|i| i + 1)
        .or_else(|| existing.starts_with("## INBOX").then_some(0));
    if let Some(at) = inbox_at {
        let mut out = String::with_capacity(existing.len() + region.len() + 1);
        out.push_str(&existing[..at]);
        out.push_str(region);
        out.push('\n');
        out.push_str(&existing[at..]);
        return (out, "inserted-before-inbox");
    }
    let mut out = String::from(existing);
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    if !out.is_empty() && !out.ends_with("\n\n") {
        out.push('\n');
    }
    out.push_str(region);
    (out, "appended")
}

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("sorter: error: {msg}");
    std::process::exit(1);
}

fn open_rt(state: &Path) -> Runtime<Event> {
    Runtime::open_described::<Store>(
        state.join("proj"),
        state.join("votes.jsonl"),
        None,
        reduce,
    )
    .unwrap_or_else(|e| fail(format!("open state {}: {e}", state.display())))
}

fn load_scope(rt: &Runtime<Event>, scope: &str) -> ScopeData {
    let db = rt.db();
    let s = Store::root().scopes().key(&scope.to_string());
    let meta: ScopeMeta = s
        .meta()
        .get(db)
        .unwrap_or_else(|e| fail(format!("read meta: {e}")))
        .unwrap_or_else(|| fail(format!("scope '{scope}' not initialized (run init)")));
    let constitution: String = s
        .constitution()
        .get(db)
        .unwrap_or_else(|e| fail(format!("read constitution: {e}")))
        .unwrap_or_default();
    let mut items = Vec::with_capacity(meta.count as usize);
    for i in 0..meta.count {
        match s.items().get(db, &i) {
            Ok(Some(t)) => items.push(t),
            Ok(None) => fail(format!("missing item {i} (store corrupt: rebuild)")),
            Err(e) => fail(format!("read item {i}: {e}")),
        }
    }
    ScopeData {
        meta,
        constitution,
        items,
    }
}

struct ScopeData {
    meta: ScopeMeta,
    constitution: String,
    items: Vec<String>,
}

fn voted_set(rt: &Runtime<Event>, scope: &str) -> HashSet<(u32, u32)> {
    let s = Store::root().scopes().key(&scope.to_string());
    s.voted()
        .keys(rt.db())
        .unwrap_or_else(|e| fail(format!("read voted pairs: {e}")))
        .into_iter()
        .collect()
}

fn pending_list(rt: &Runtime<Event>, scope: &str) -> Vec<(String, Pending)> {
    let s = Store::root().scopes().key(&scope.to_string());
    s.pending()
        .iter(rt.db())
        .unwrap_or_else(|e| fail(format!("read pending: {e}")))
}

fn vote_count(rt: &Runtime<Event>, scope: &str) -> i64 {
    let s = Store::root().scopes().key(&scope.to_string());
    s.count().get(rt.db()).unwrap_or_else(|e| fail(format!("read count: {e}")))
}

/// Random-walk scores over the WHOLE edge graph. Order follows item indices.
/// Only meaningful when connected — across components the numbers share a
/// denominator but no comparisons, so treat global scores as rough weather.
fn scores(rt: &Runtime<Event>, scope: &str, n: usize) -> Vec<f64> {
    let s = Store::root().scopes().key(&scope.to_string());
    let edges = s
        .edges()
        .iter(rt.db())
        .unwrap_or_else(|e| fail(format!("read edges: {e}")));
    let iter = edges.into_iter().map(|((a, b), w)| ((a as usize, b as usize), w));
    ranking::compute_scores_from_edges(n, iter, ranking::MAX_ITERS, ranking::TOL)
}

/// Per-component rankings: the honest output. Each connected hunk is ranked
/// on its own induced subgraph (own denominator, own scale), biggest group
/// first. Global scores ride along for weather, not for rank.
fn grouped_ranking(
    rt: &Runtime<Event>,
    scope: &str,
    n: usize,
) -> (Vec<f64>, Vec<ranking::RankGroup>) {
    let s = Store::root().scopes().key(&scope.to_string());
    let edges = s
        .edges()
        .iter(rt.db())
        .unwrap_or_else(|e| fail(format!("read edges: {e}")));
    let triples: Vec<(usize, usize, f64)> = edges
        .into_iter()
        .map(|((a, b), w)| (a as usize, b as usize, w))
        .collect();
    let global = ranking::compute_scores_from_edges(
        n,
        triples.iter().map(|&(a, b, w)| ((a, b), w)),
        ranking::MAX_ITERS,
        ranking::TOL,
    );
    let voted_usize: std::collections::HashSet<(usize, usize)> = voted_set(rt, scope)
        .into_iter()
        .map(|(a, b)| (a as usize, b as usize))
        .collect();
    let groups = ranking::rank_groups(n, &voted_usize, &triples, ranking::MAX_ITERS, ranking::TOL);
    (global, groups)
}

/// The renderable ranking for MEMORY.md: ONLY the largest connected
/// component, ranked on its own induced subgraph. Unvoted items are
/// undefined — they belong to no ranking and never render; smaller
/// components wait for the dealer to bridge them in. Returns the
/// (index, score) list, best-first, truncated to `top`, plus the size of
/// the ranked universe. A largest component under 2 members means no
/// vote has landed yet: the ranking is empty and the universe is 0.
fn largest_group_ranked(groups: &[ranking::RankGroup], top: usize) -> (Vec<(u32, f64)>, usize) {
    match groups.first() {
        Some(g) if g.members.len() >= 2 => {
            let ranked = g
                .ranked
                .iter()
                .take(top)
                .map(|&(m, s)| (m as u32, s))
                .collect();
            (ranked, g.members.len())
        }
        _ => (Vec::new(), 0),
    }
}

fn progress_json(rt: &Runtime<Event>, scope: &str, n: usize) -> serde_json::Value {
    let votes = vote_count(rt, scope);
    let voted = voted_set(rt, scope);
    let pending = pending_list(rt, scope).len() as u64;
    let total = (n as u64) * (n as u64).saturating_sub(1) / 2;
    // Votes until connected: every bridging vote merges two components, so
    // components - 1 is the exact count of right deals left. Zero means the
    // whole scope ranks on one scale — the alphabet oracle's O(n) promise.
    let comps = pair::component_ids(n as u32, &voted);
    let components = comps.iter().copied().max().map(|m| m + 1).unwrap_or(if n == 0 { 0 } else { 1 });
    let largest = {
        let mut sizes = vec![0u64; components.max(1)];
        for &c in &comps {
            sizes[c] += 1;
        }
        sizes.into_iter().max().unwrap_or(0)
    };
    serde_json::json!({
        "votes": votes,
        "voted_pairs": voted.len() as u64,
        "pending": pending,
        "total_pairs": total,
        "components": components,
        "largest_component": largest,
        "votes_until_connected": components.saturating_sub(1),
        "connected": components <= 1,
    })
}
/// Load KEY=VAL lines from a file, `#` comments skipped. Real env wins,
/// same collar as hopd's load_dotenv — never prints values.
fn load_env_file(path: &Path) {
    let Ok(text) = fs::read_to_string(path) else {
        fail(format!("read env file {}: missing or unreadable", path.display()))
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else { continue };
        let (k, v) = (k.trim(), v.trim());
        // Strip one layer of matching quotes, hopd-style files quote secrets.
        let v = v.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(
            v.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')).unwrap_or(v),
        );
        if !k.is_empty() && std::env::var_os(k).is_none() {
            // SAFETY: single-threaded at startup before any spawn; same as hopd.
            unsafe { std::env::set_var(k, v) };
        }
    }
}

/// One judgment from the OpenRouter judge: returns (a_share, b_share, reason).
/// Refuses to print the key — it rides only in the Authorization header.
fn judge_pair(
    client: &reqwest::blocking::Client,
    base_url: &str,
    key: &str,
    model: &str,
    constitution: &str,
    cmp_id: &str,
    a_text: &str,
    b_text: &str,
) -> Result<(f64, f64, String), String> {
    let system = format!(
        "You judge pairwise memories under this declaration:\n\n{constitution}\n\nReply with a JSON object only: {{\"ratio_a\": <number>, \"ratio_b\": <number>, \"reason\": \"<why the higher share earns it, citing declaration laws>\"}}. Rules: ratio_a pours to item A, ratio_b to item B; both strictly > 0, never equal — a tie names no winner. The reason must argue under the declaration and cite its laws."
    );
    let user = format!("comparison {cmp_id}\n\nA:\n{a_text}\n\nB:\n{b_text}");
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    });
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {key}"))
        .header("HTTP-Referer", "https://github.com/sorter")
        .header("X-Title", "sorter vote-loop")
        .json(&body)
        .send()
        .map_err(|e| format!("judge request: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("judge http {}", resp.status()));
    }
    let v: serde_json::Value = resp.json().map_err(|e| format!("judge decode: {e}"))?;
    let content = v
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
        .ok_or_else(|| "judge: empty choices".to_string())?;
    let j: serde_json::Value =
        serde_json::from_str(content).map_err(|e| format!("judge json: {e}: {content}"))?;
    let ra = j.get("ratio_a").and_then(|x| x.as_f64()).ok_or_else(|| format!("judge: bad ratio_a in {content}"))?;
    let rb = j.get("ratio_b").and_then(|x| x.as_f64()).ok_or_else(|| format!("judge: bad ratio_b in {content}"))?;
    let reason = j
        .get("reason")
        .and_then(|x| x.as_str())
        .ok_or_else(|| format!("judge: missing reason in {content}"))?
        .to_string();
    if !(ra.is_finite() && rb.is_finite()) || ra <= 0.0 || rb <= 0.0 {
        return Err(format!("judge: shares must both be > 0, got {ra}:{rb}"));
    }
    if ra == rb {
        return Err(format!("judge: tie {ra}:{rb} names no winner"));
    }
    if reason.trim().is_empty() {
        return Err("judge: empty reason".to_string());
    }
    Ok((ra, rb, reason))
}

/// Deal one open pair with the Next-without---seed heart: bridging-first,
/// degree-spread, current-overlap-shy. None = nothing left to chew.
fn deal_open_pair(rt: &Runtime<Event>, scope: &str, n: u32) -> Option<(u32, u32)> {
    let voted = voted_set(rt, scope);
    let pendings = pending_list(rt, scope);
    let pending_pairs: HashSet<(u32, u32)> = pendings
        .iter()
        .map(|(_, p)| if p.a < p.b { (p.a, p.b) } else { (p.b, p.a) })
        .collect();
    let current: Option<(u32, u32)> = pendings
        .iter()
        .map(|(_, p)| (p.a, p.b))
        .max_by_key(|(a, b)| (*a as u64) << 32 | *b as u64)
        .map(|(a, b)| if a <= b { (a, b) } else { (b, a) });
    pair::suggest_next_pair(n, &voted, &pending_pairs, current, None)
}


fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Init {
            memories,
            format,
            feed,
            constitution,
            scope,
            force,
        } => {
            if scope.contains('/') || scope.is_empty() {
                fail("scope must be a plain name");
            }
            if cli.state.join("votes.jsonl").exists() || cli.state.join("proj").exists() {
                if !force {
                    fail(format!(
                        "state {} exists (use --force to wipe)",
                        cli.state.display()
                    ));
                }
                fs::remove_dir_all(&cli.state)
                    .unwrap_or_else(|e| fail(format!("wipe state: {e}")));
            }
            let (raw, items) = match (&memories, &feed) {
                (Some(mem), None) => {
                    let src = fs::read_to_string(mem)
                        .unwrap_or_else(|e| fail(format!("read {}: {e}", mem.display())));
                    let fmt = match format.as_deref() {
                        Some(f) => f.to_string(),
                        None => match mem.extension().and_then(|e| e.to_str()) {
                            Some("jsonl") => "jsonl".to_string(),
                            _ => "md".to_string(),
                        },
                    };
                    let items = match fmt.as_str() {
                        "md" => {
                            let items = parse_memories(&src);
                            if items.is_empty() {
                                fail("no top-level list items found in memories file");
                            }
                            items
                        }
                        "jsonl" => {
                            let items = parse_memories_jsonl(&src).unwrap_or_else(|e| {
                                fail(format!("parse {}: {e}", mem.display()))
                            });
                            if items.is_empty() {
                                fail("no memories found in jsonl file");
                            }
                            items
                        }
                        other => fail(format!("unknown --format '{other}': use md or jsonl")),
                    };
                    (src, items)
                }
                (None, Some(feed)) => {
                    let src = fs::read_to_string(feed)
                        .unwrap_or_else(|e| fail(format!("read {}: {e}", feed.display())));
                    let items = parse_feed(&src);
                    if items.is_empty() {
                        fail("no votable turns found in feed file");
                    }
                    (src, items)
                }
                _ => fail("give exactly one of --memories or --feed"),
            };
            let con = fs::read_to_string(&constitution)
                .unwrap_or_else(|e| fail(format!("read {}: {e}", constitution.display())));
            if con.trim().is_empty() {
                fail("constitution is empty");
            }
            let rt = open_rt(&cli.state);
            rt.append(Event::Init {
                scope: scope.clone(),
                memories_hash: fingerprint(&raw),
                constitution_hash: fingerprint(&con),
                items: items.clone(),
                constitution: con.clone(),
            })
            .unwrap_or_else(|e| fail(format!("append init: {e}")));
            println!(
                "{}",
                serde_json::json!({
                    "ok": true,
                    "scope": scope,
                    "items": items.len(),
                    "constitution_chars": con.len(),
                    "constitution_hash": fingerprint(&con),
                    "state": cli.state.display().to_string(),
                })
            );
        }
        Cmd::Next { scope, seed, a, b } => {
            let rt = open_rt(&cli.state);
            let data = load_scope(&rt, &scope);
            let n = data.items.len() as u32;
            if n < 2 {
                fail("need at least 2 memories to deal a pair");
            }
            // mem-0012 or 12 — stable ids from ranking output, raw index, done.
            let parse_item = |s: &str| -> u32 {
                let t = s.strip_prefix("mem-").unwrap_or(s);
                t.parse::<u32>().unwrap_or_else(|_| {
                    fail(format!("bad item '{s}': use mem-0012 or 12 (see ranking ids)"))
                })
            };
            let voted = voted_set(&rt, &scope);
            let pendings = pending_list(&rt, &scope);
            let pending_pairs: HashSet<(u32, u32)> = pendings
                .iter()
                .map(|(_, p)| if p.a < p.b { (p.a, p.b) } else { (p.b, p.a) })
                .collect();
            // Explicit-pair gate lives in pair.rs now (Master: no one-offs).
            let deal_open = |x: u32, y: u32| pair::deal_open_pair(&voted, &pending_pairs, x, y, n);
            // Explicit pair beats everything: Master points, I spread.
            // Single pin: best partner for her, bridging-first.
            // Neither: the heart decides (or the seed, for tests).
            let dealt: Option<(u32, u32)> = match (a, b) {
                (Some(a_s), Some(b_s)) => {
                    let (x, y) = (parse_item(&a_s), parse_item(&b_s));
                    if x >= n || y >= n {
                        fail(format!("item out of range (0..{n}): {a_s} {b_s}"));
                    }
                    if x == y {
                        fail("cannot deal an item against itself");
                    }
                    if !deal_open(x, y) {
                        fail("that pair is already voted or pending — pick another");
                    }
                    Some((x, y))
                }
                (Some(a_s), None) => {
                    let x = parse_item(&a_s);
                    if x >= n {
                        fail(format!("item out of range (0..{n}): {a_s}"));
                    }
                    if seed.is_some() {
                        fail("--seed with --a needs no partner search; drop --seed");
                    }
                    pair::suggest_partner_for(x, n, &voted, &pending_pairs).map(|y| (x, y))
                }
                (None, Some(_)) => fail("--b requires --a (pin the left side first)"),
                (None, None) => {
                    // Overlap-shy: don't re-deal items from the most recent deal
                    // still awaiting judgment, so the agent sees fresh meat first.
                    let current: Option<(u32, u32)> = pendings
                        .iter()
                        .map(|(_, p)| (p.a, p.b))
                        .max_by_key(|(a, b)| (*a as u64) << 32 | *b as u64)
                        .map(|(a, b)| if a <= b { (a, b) } else { (b, a) });
                    match seed {
                        // Deterministic escape hatch for tests: uniform over opens.
                        Some(s) => {
                            let mut candidates = Vec::new();
                            for a in 0..n {
                                for b in (a + 1)..n {
                                    if !voted.contains(&(a, b)) && !pending_pairs.contains(&(a, b)) {
                                        candidates.push((a, b));
                                    }
                                }
                            }
                            if candidates.is_empty() {
                                None
                            } else {
                                use rand::seq::IndexedRandom;
                                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(s);
                                Some(*candidates.choose(&mut rng).unwrap())
                            }
                        }
                        // scope_rank's heart: bridging-first, degree-spread, overlap-shy.
                        None => pair::suggest_next_pair(n, &voted, &pending_pairs, current, None),
                    }
                }
            };
            let Some((a, b)) = dealt else {
                println!(
                    "{}",
                    serde_json::json!({
                        "done": true,
                        "scope": scope,
                        "progress": progress_json(&rt, &scope, n as usize),
                    })
                );
                return;
            };
            // cmp id is the tape seq: stable across replay, never collides.
            let seq = rt.log_len().unwrap_or_else(|e| fail(format!("read log len: {e}")));
            let cmp_id = format!("cmp-{seq:04}");
            rt.append(Event::Deal {
                scope: scope.clone(),
                cmp_id: cmp_id.clone(),
                a,
                b,
            })
            .unwrap_or_else(|e| fail(format!("append deal: {e}")));
            println!(
                "{}",
                serde_json::json!({
                    "comparison_id": cmp_id,
                    "scope": scope,
                    "a": {"id": mem_id(a), "index": a, "text": data.items[a as usize]},
                    "b": {"id": mem_id(b), "index": b, "text": data.items[b as usize]},
                    "constitution": data.constitution,
                    "constitution_hash": data.meta.constitution_hash,
                    "progress": progress_json(&rt, &scope, n as usize),
                })
            );
        }
        Cmd::Vote {
            cmp_id,
            ratio,
            reason,
            constitution_hash,
            weight,
            scope,
        } => {
            if reason.trim().is_empty() {
                fail("reason is required: say why the higher share earns it under the constitution");
            }
            if !weight.is_finite() || weight <= 0.0 {
                fail("weight must be a positive number");
            }
            // Scalar, always: "10:7" pours 10 to a, 7 to b. Both shares
            // strictly positive, never equal — a tie names no winner, so it
            // names nothing. Zero starves an edge and collapses the walk's
            // normalization. No alias, no default: the agent's mouth or death.
            let (a_share, b_share): (f64, f64) = {
                let (x, y) = ratio.split_once(':').unwrap_or_else(|| {
                    fail("ratio must look like 10:7 (first number is for a, second for b)")
                });
                let x: f64 = x.trim().parse().unwrap_or_else(|_| fail("ratio shares must be numbers > 0, like 10:7"));
                let y: f64 = y.trim().parse().unwrap_or_else(|_| fail("ratio shares must be numbers > 0, like 10:7"));
                if !(x.is_finite() && y.is_finite()) || x <= 0.0 || y <= 0.0 {
                    fail("ratio shares must both be > 0 (no zeroes, no shutouts — every side earns its edge)");
                }
                if x == y {
                    fail(format!("ratio {x}:{y} is a tie — it names no winner. Say which side earns more under the constitution."));
                }
                (x, y)
            };
            let rt = open_rt(&cli.state);
            let data = load_scope(&rt, &scope);
            // The lock Master kissed in: no hash, no vote. A stale agent
            // holding yesterday's constitution dies here, must re-read her.
            if constitution_hash != data.meta.constitution_hash {
                fail(format!(
                    "stale constitution: you hold {constitution_hash}, scope wants {} (re-run next)",
                    data.meta.constitution_hash
                ));
            }
            let p = pending_list(&rt, &scope)
                .into_iter()
                .find(|(id, _)| id == &cmp_id)
                .map(|(_, p)| p)
                .unwrap_or_else(|| fail(format!("unknown or already-judged {cmp_id}")));
            // a_share pours to a, b_share to b — winner is derived, not chosen.
            let (wi, li) = if a_share >= b_share { (p.a, p.b) } else { (p.b, p.a) };
            rt.append(Event::Vote {
                scope: scope.clone(),
                cmp_id: cmp_id.clone(),
                a_item: p.a,
                b_item: p.b,
                a_share,
                b_share,
                winner: wi,
                loser: li,
                weight,
                reason: reason.clone(),
                constitution_hash: constitution_hash.clone(),
            })
            .unwrap_or_else(|e| fail(format!("append vote: {e}")));
            let sc = scores(&rt, &scope, data.items.len());
            let at = |i: u32| *sc.get(i as usize).unwrap_or(&0.0);
            println!(
                "{}",
                serde_json::json!({
                    "ok": true,
                    "comparison_id": cmp_id,
                    "ratio": format!("{a_share}:{b_share}"),
                    "winner": {"id": mem_id(wi), "text": data.items[wi as usize], "score": at(wi)},
                    "loser": {"id": mem_id(li), "text": data.items[li as usize], "score": at(li)},
                    "weight": weight,
                    "total_votes": vote_count(&rt, &scope),
                })
            );
        }
        Cmd::Ranking { scope, top } => {
            let rt = open_rt(&cli.state);
            let data = load_scope(&rt, &scope);
            let n = data.items.len();
            let (_global, groups) = grouped_ranking(&rt, &scope, n);
            // No flat view: items only exist inside their groups, each ranked
            // on its own wet scale. Biggest hunk first; `--top` caps groups.
            let groups_json: Vec<_> = groups
                .iter()
                .enumerate()
                .take(top)
                .map(|(gi, g)| {
                    serde_json::json!({
                        "group": gi,
                        "size": g.members.len(),
                        "pairs": g.pairs,
                        "ranked": g.ranked.iter().enumerate().map(|(pos, &(m, s))| {
                            serde_json::json!({
                                "group_rank": pos + 1,
                                "id": mem_id(m as u32),
                                "index": m,
                                "score": s,
                                "text": data.items[m],
                            })
                        }).collect::<Vec<_>>(),
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::json!({
                    "scope": scope,
                    "votes": vote_count(&rt, &scope),
                    "connected": groups.len() <= 1,
                    "groups": groups_json,
                })
            );
        }
        Cmd::Render { scope, top, output } => {
            let rt = open_rt(&cli.state);
            let data = load_scope(&rt, &scope);
            let (_global, groups) = grouped_ranking(&rt, &scope, data.items.len());
            let votes = vote_count(&rt, &scope);
            let mut md = format!(
                "# top memories — {scope} ({votes} votes, {} groups)\n\n",
                groups.len(),
            );
            for (gi, g) in groups.iter().take(top).enumerate() {
                md.push_str(&format!("## group {gi} — {} items, {} pairs\n\n", g.members.len(), g.pairs));
                for (pos, &(m, s)) in g.ranked.iter().enumerate() {
                    md.push_str(&format!("### {}. {} · score {s:.4}\n\n{}\n\n", pos + 1, mem_id(m as u32), data.items[m]));
                }
            }
            fs::write(&output, md).unwrap_or_else(|e| fail(format!("write {}: {e}", output.display())));
            println!(
                "{}",
                serde_json::json!({"ok": true, "output": output.display().to_string(), "votes": votes})
            );
        }
        Cmd::RenderMemory { scope, top, input, output, heading } => {
            let rt = open_rt(&cli.state);
            let data = load_scope(&rt, &scope);
            // The honest ranking: largest connected component only, on its
            // own induced subgraph. Global whole-pool scores are weather,
            // not rank — unvoted items are undefined and never render.
            let (_global, groups) = grouped_ranking(&rt, &scope, data.items.len());
            let (ranked, universe) = largest_group_ranked(&groups, top);
            let votes = vote_count(&rt, &scope);
            let region = render_memory_region(&scope, &heading, &ranked, &data.items, votes, universe);
            let out_path = output
                .clone()
                .or_else(|| input.clone())
                .unwrap_or_else(|| PathBuf::from("MEMORY.md"));
            let existing = input.as_ref().and_then(|p| fs::read_to_string(p).ok());
            let (text, mode) = match &existing {
                Some(ex) if !ex.trim().is_empty() => merge_memory_md(ex, &region, &scope),
                _ => (
                    format!("# MEMORY.md\n\n{region}\n## INBOX — unranked claims tray (append-only; folded into rank in session)\n"),
                    "created",
                ),
            };
            // Atomic write: temp file in the same directory, then rename over.
            let tmp = out_path.with_extension("md.tmp-sorter");
            fs::write(&tmp, &text)
                .unwrap_or_else(|e| fail(format!("write {}: {e}", tmp.display())));
            fs::rename(&tmp, &out_path)
                .unwrap_or_else(|e| fail(format!("rename to {}: {e}", out_path.display())));
            println!(
                "{}",
                serde_json::json!({"ok": true, "output": out_path.display().to_string(), "scope": scope, "votes": votes, "items_written": ranked.len(), "ranked_universe": universe, "components": groups.len(), "mode": mode})
            );
        }
        Cmd::Status { scope } => {
            let rt = open_rt(&cli.state);
            let data = load_scope(&rt, &scope);
            let mut p = progress_json(&rt, &scope, data.items.len());
            p["scope"] = serde_json::json!(scope);
            p["items"] = serde_json::json!(data.items.len());
            p["memories_hash"] = serde_json::json!(data.meta.memories_hash);
            p["constitution_hash"] = serde_json::json!(data.meta.constitution_hash);
            p["pending_ids"] = serde_json::json!(
                pending_list(&rt, &scope).into_iter().map(|(id, _)| id).collect::<Vec<_>>()
            );
            println!("{p}");
        }
        Cmd::Log { scope, last } => {
            let rt = open_rt(&cli.state);
            let _ = load_scope(&rt, &scope);
            let len = rt.log_len().unwrap_or_else(|e| fail(format!("read log len: {e}")));
            let start = len.saturating_sub(last as u64 * 4 + 16);
            let recs = rt
                .records_from(start)
                .unwrap_or_else(|e| fail(format!("read tape: {e}")));
            let mut out = Vec::new();
            for r in recs {
                if let Event::Vote {
                    scope: s,
                    cmp_id,
                    a_item: _,
                    b_item: _,
                    a_share,
                    b_share,
                    winner,
                    loser,
                    weight,
                    reason,
                    constitution_hash: _,
                } = r.event
                {
                    if s != scope {
                        continue;
                    }
                    out.push(serde_json::json!({
                        "seq": r.seq,
                        "comparison_id": cmp_id,
                        "ratio": format!("{a_share}:{b_share}"),
                        "winner": mem_id(winner),
                        "loser": mem_id(loser),
                        "weight": weight,
                        "reason": reason,
                    }));
                }
            }
            let tail: Vec<_> = out.into_iter().rev().take(last).rev().collect();
            println!("{}", serde_json::json!(tail));
        }
        Cmd::Show { cmp_id, scope } => {
            let rt = open_rt(&cli.state);
            let data = load_scope(&rt, &scope);
            // Pending first; else recover from the tape (already judged).
            let found = pending_list(&rt, &scope)
                .into_iter()
                .find(|(id, _)| id == &cmp_id)
                .map(|(_, p)| (p.a, p.b));
            let (a, b) = match found {
                Some(ab) => ab,
                None => {
                    let len = rt.log_len().unwrap_or_else(|e| fail(format!("read log len: {e}")));
                    let recs = rt
                        .records_from(0)
                        .unwrap_or_else(|e| fail(format!("read tape: {e}")));
                    let _ = len;
                    recs
                        .into_iter()
                        .filter_map(|r| match r.event {
                            Event::Deal { scope: s, cmp_id: c, a, b } if s == scope && c == cmp_id => {
                                Some((a, b))
                            }
                            _ => None,
                        })
                        .next()
                        .unwrap_or_else(|| fail(format!("unknown {cmp_id}")))
                }
            };
            println!(
                "{}",
                serde_json::json!({
                    "comparison_id": cmp_id,
                    "scope": scope,
                    "a": {"id": mem_id(a), "index": a, "text": data.items[a as usize]},
                    "b": {"id": mem_id(b), "index": b, "text": data.items[b as usize]},
                    "constitution": data.constitution,
                })
            );
        }
        Cmd::VoteLoop { scope, model, base_url, key_var, env_file, votes: want, weight, timeout } => {
            if !weight.is_finite() || weight <= 0.0 {
                fail("weight must be a positive number");
            }
            if let Some(p) = &env_file {
                load_env_file(p);
            } else {
                // Matches hopd: cwd ./.env feeds process env, real env wins.
                let _ = dotenvy::dotenv();
            }
            let key = std::env::var(&key_var).unwrap_or_else(|_| {
                fail(format!("{key_var} is not set (pass --env-file or export it)"))
            });
            if key.trim().is_empty() {
                fail(format!("{key_var} is empty"));
            }
            let client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(timeout.max(1)))
                .build()
                .unwrap_or_else(|e| fail(format!("http client: {e}")));
            let rt = open_rt(&cli.state);
            let data = load_scope(&rt, &scope);
            let n = data.items.len() as u32;
            if n < 2 {
                fail("need at least 2 memories to deal a pair");
            }
            let target = if want == 0 { u64::MAX } else { want };
            let already = vote_count(&rt, &scope).max(0) as u64;
            let t0 = std::time::Instant::now();
            let mut done: u64 = 0;
            let mut durs: Vec<f64> = Vec::new();
            // Chew: deal one, beg the judge, swallow her ratio onto the tape.
            while done < target {
                let Some((a, b)) = deal_open_pair(&rt, &scope, n) else { break };
                let seq = rt.log_len().unwrap_or_else(|e| fail(format!("read log len: {e}")));
                let cmp_id = format!("cmp-{seq:04}");
                rt.append(Event::Deal { scope: scope.clone(), cmp_id: cmp_id.clone(), a, b })
                    .unwrap_or_else(|e| fail(format!("append deal: {e}")));
                let vt0 = std::time::Instant::now();
                let (ra, rb, reason) = match judge_pair(
                    &client, &base_url, &key, &model,
                    &data.constitution, &cmp_id,
                    &data.items[a as usize], &data.items[b as usize],
                ) {
                    Ok(j) => j,
                    Err(e) => fail(format!("{cmp_id} judge failed: {e}")),
                };
                let (wi, li) = if ra >= rb { (a, b) } else { (b, a) };
                rt.append(Event::Vote {
                    scope: scope.clone(), cmp_id: cmp_id.clone(),
                    a_item: a, b_item: b, a_share: ra, b_share: rb,
                    winner: wi, loser: li, weight,
                    reason, constitution_hash: data.meta.constitution_hash.clone(),
                })
                .unwrap_or_else(|e| fail(format!("append vote: {e}")));
                let dt = vt0.elapsed().as_secs_f64();
                durs.push(dt);
                done += 1;
                let sc = scores(&rt, &scope, n as usize);
                let at = |i: u32| *sc.get(i as usize).unwrap_or(&0.0);
                println!(
                    "{}",
                    serde_json::json!({
                        "ok": true,
                        "comparison_id": cmp_id,
                        "ratio": format!("{ra}:{rb}"),
                        "winner": {"id": mem_id(wi), "text": data.items[wi as usize], "score": at(wi)},
                        "loser": {"id": mem_id(li), "text": data.items[li as usize], "score": at(li)},
                        "weight": weight,
                        "total_votes": vote_count(&rt, &scope),
                    })
                );
                // Tqdm-style drip on stderr: Master watches the hole fill.
                let total_disp = if want == 0 { "until done".to_string() } else { want.to_string() };
                let avg = durs.iter().sum::<f64>() / durs.len().max(1) as f64;
                let elapsed = t0.elapsed().as_secs_f64();
                let prog = progress_json(&rt, &scope, n as usize);
                let vuc = prog.get("votes_until_connected").and_then(|v| v.as_u64()).unwrap_or(0);
                let eta_s = avg * (if want == 0 { vuc } else { target.saturating_sub(done) }) as f64;
                let (em, es) = ((eta_s / 60.0) as u64, (eta_s % 60.0) as u64);
                eprintln!(
                    "[{}/{}] {} a:{} b:{} ratio {ra}:{rb} elapsed {:.0}s avg {:.1}s/vote ETA {:02}:{:02} votes_until_connected {vuc}",
                    already + done, total_disp, cmp_id, mem_id(a), mem_id(b), elapsed, avg, em, es
                );
                // Keep stderr wet and immediate: flush every line.
                use std::io::Write as _;
                let _ = std::io::stderr().flush();
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let avg = if durs.is_empty() { 0.0 } else { durs.iter().sum::<f64>() / durs.len() as f64 };
            let prog = progress_json(&rt, &scope, n as usize);
            let connected = prog.get("connected").and_then(|v| v.as_bool()).unwrap_or(false);
            eprintln!(
                "vote-loop done: {done} votes, elapsed {:.1}s, avg {:.1}s/vote, connected {connected}",
                elapsed, avg
            );
        }
        Cmd::Verify => {
            let rt = open_rt(&cli.state);
            rt.verify().unwrap_or_else(|e| fail(format!("verify: {e}")));
            println!("{}", serde_json::json!({"ok": true, "replay": "incremental == from zero"}));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_opinions_into_200_raw_memories() {
        let src = std::fs::read_to_string("data/opinions.md").expect("test data");
        let items = parse_memories(&src);
        let raw: Vec<&str> = src.lines().filter(|l| l.starts_with("- ")).collect();
        assert_eq!(items.len(), 200);
        assert_eq!(items.len(), raw.len());
        // Raw slices: first 60 alnum chars identical to the source lines.
        let alnum = |s: &str| s.chars().filter(|c| c.is_alphanumeric()).take(60).collect::<String>();
        for (parsed, line) in items.iter().zip(raw.iter()) {
            assert_eq!(alnum(parsed), alnum(&line[2..]));
        }
    }

    #[test]
    fn jsonl_preserves_bodies_byte_exact() {
        let src = concat!(
            "{\"ts_pt\": \"2026-09-17 18:34\", \"body\": \":) keep breathing.\\n\\nsecond para  \"}\n",
            "{\"ts_pt\": \"2026-09-17 18:36\", \"body\": \"- not a list item, just my words\"}\n",
            "{\"body\": \"no timestamp, still mine — curly ’ quotes\"}\n",
            "\n",
            "{\"ts_pt\": \"2026-09-17 18:38\", \"body\": \"   leading spaces kept\"}\n"
        );
        let items = parse_memories_jsonl(src).expect("valid jsonl");
        assert_eq!(items.len(), 4);
        assert_eq!(items[0], "[2026-09-17 18:34 PT] :) keep breathing.\n\nsecond para  ");
        assert_eq!(items[1], "[2026-09-17 18:36 PT] - not a list item, just my words");
        assert_eq!(items[2], "no timestamp, still mine — curly ’ quotes");
        assert_eq!(items[3], "[2026-09-17 18:38 PT]    leading spaces kept");
    }

    #[test]
    fn jsonl_rejects_malformed_lines_loudly() {
        let err = parse_memories_jsonl("{\"body\": \"fine\"}\nnot json\n").unwrap_err();
        assert!(err.contains("line 2"), "got: {err}");
        let err = parse_memories_jsonl("{\"ts_pt\": \"2026-01-01 00:00\"}\n").unwrap_err();
        assert!(err.contains("body"), "got: {err}");
    }

    #[test]
    fn ignores_blockquote_and_heading() {
        let items = parse_memories("> quote\n\n# head\n\nplain\n\n- a\n- b\n");
        assert_eq!(items, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn nested_lists_stay_whole() {
        let items = parse_memories("- outer\n  - inner\n- second\n");
        assert_eq!(items.len(), 2);
        assert!(items[0].contains("inner"));
    }

    fn sample_region(scope: &str, note: &str) -> String {
        let items = vec![format!("alpha {note}"), "beta".to_string()];
        render_memory_region(scope, "Ranked working set", &[(0, 0.6), (1, 0.4)], &items, 7, 2)
    }

    fn fake_group(members: &[usize], ranked: &[(usize, f64)]) -> ranking::RankGroup {
        ranking::RankGroup {
            members: members.to_vec(),
            ranked: ranked.to_vec(),
            pairs: ranked.len(),
        }
    }

    #[test]
    fn render_ranking_is_largest_component_only() {
        // Biggest group first (rank_groups' contract); a smaller voted
        // component and singletons must never leak into the render.
        let groups = vec![
            fake_group(&[3, 1, 9], &[(3, 0.5), (1, 0.3), (9, 0.2)]),
            fake_group(&[0, 2], &[(0, 0.7), (2, 0.3)]),
            fake_group(&[4], &[(4, 1.0)]),
        ];
        let (ranked, universe) = largest_group_ranked(&groups, 20);
        assert_eq!(universe, 3);
        assert_eq!(ranked, vec![(3, 0.5), (1, 0.3), (9, 0.2)]);
        let (top2, _) = largest_group_ranked(&groups, 2);
        assert_eq!(top2, vec![(3, 0.5), (1, 0.3)]);
    }

    #[test]
    fn render_ranking_undefined_until_a_vote_lands() {
        // All singletons (nothing voted): no ranking, universe 0 — an
        // unvoted item must never be seated as "rank 1".
        let groups = vec![fake_group(&[0], &[(0, 1.0)]), fake_group(&[1], &[(1, 1.0)])];
        let (ranked, universe) = largest_group_ranked(&groups, 20);
        assert!(ranked.is_empty());
        assert_eq!(universe, 0);
        let (ranked, universe) = largest_group_ranked(&[], 20);
        assert!(ranked.is_empty());
        assert_eq!(universe, 0);
    }

    #[test]
    fn memory_region_carries_provenance() {
        let r = sample_region("world", "x");
        assert!(r.starts_with("<!-- sorter:begin scope=world -->\n"));
        assert!(r.trim_end().ends_with("<!-- sorter:end -->"));
        assert!(r.contains("- alpha x <!-- sorter:rank 1 score 0.6000 id mem-0000 -->"));
        assert!(r.contains("- beta <!-- sorter:rank 2 score 0.4000 id mem-0001 -->"));
    }

    #[test]
    fn merge_inserts_before_inbox_and_preserves_the_rest() {
        let doc = "# MEMORY.md\n\n<!-- treaty -->\n\n## Facts\n- hand written\n\n## INBOX — tray\n<!-- land here -->\n";
        let (out, mode) = merge_memory_md(doc, &sample_region("world", "x"), "world");
        assert_eq!(mode, "inserted-before-inbox");
        // Everything hand-written survives, in order, INBOX still last.
        assert!(out.contains("<!-- treaty -->"));
        assert!(out.contains("## Facts\n- hand written\n"));
        assert!(out.find("sorter:begin").unwrap() < out.find("## INBOX").unwrap());
        assert!(out.ends_with("<!-- land here -->\n"));
    }

    #[test]
    fn merge_replace_is_idempotent() {
        let doc = "# MEMORY.md\n\n## INBOX — tray\n";
        let region = sample_region("world", "x");
        let (once, _) = merge_memory_md(doc, &region, "world");
        let (twice, mode) = merge_memory_md(&once, &region, "world");
        assert_eq!(mode, "replaced");
        assert_eq!(once, twice);
        // A changed region replaces in place; the rest of the file is unmoved.
        let (thrice, _) = merge_memory_md(&once, &sample_region("world", "y"), "world");
        assert!(thrice.contains("alpha y"));
        assert!(!thrice.contains("alpha x"));
        assert!(thrice.starts_with("# MEMORY.md\n"));
        assert!(thrice.ends_with("## INBOX — tray\n"));
    }

    #[test]
    fn merge_scopes_coexist() {
        let doc = "# MEMORY.md\n\n## INBOX — tray\n";
        let (one, _) = merge_memory_md(doc, &sample_region("world", "x"), "world");
        let (two, _) = merge_memory_md(&one, &sample_region("opinions", "x"), "opinions");
        assert!(two.contains("scope=world"));
        assert!(two.contains("scope=opinions"));
        assert_eq!(two.matches("## INBOX").count(), 1);
    }
}
