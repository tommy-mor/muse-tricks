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
        /// Markdown file of memories (top-level list items).
        #[arg(long)]
        memories: PathBuf,
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
    Next {
        #[arg(long, default_value = "opinions")]
        scope: String,
        /// Deterministic pick (tests). Default: fresh randomness every deal.
        #[arg(long)]
        seed: Option<u64>,
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
    /// Show the current ranking.
    Ranking {
        #[arg(long, default_value = "opinions")]
        scope: String,
        #[arg(long, default_value = "20")]
        top: usize,
    },
    /// Render top memories as a markdown view file.
    Render {
        #[arg(long, default_value = "opinions")]
        scope: String,
        #[arg(long, default_value = "50")]
        top: usize,
        /// Output markdown file.
        #[arg(long, default_value = "top.md")]
        output: PathBuf,
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
    /// Prove incremental == replay from zero.
    Verify,
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

/// Random-walk scores over the edge graph. Order follows item indices.
fn scores(rt: &Runtime<Event>, scope: &str, n: usize) -> Vec<f64> {
    let s = Store::root().scopes().key(&scope.to_string());
    let edges = s
        .edges()
        .iter(rt.db())
        .unwrap_or_else(|e| fail(format!("read edges: {e}")));
    let iter = edges.into_iter().map(|((a, b), w)| ((a as usize, b as usize), w));
    ranking::compute_scores_from_edges(n, iter, ranking::MAX_ITERS, ranking::TOL)
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

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Init {
            memories,
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
            let md = fs::read_to_string(&memories)
                .unwrap_or_else(|e| fail(format!("read {}: {e}", memories.display())));
            let con = fs::read_to_string(&constitution)
                .unwrap_or_else(|e| fail(format!("read {}: {e}", constitution.display())));
            let items = parse_memories(&md);
            if items.is_empty() {
                fail("no top-level list items found in memories file");
            }
            if con.trim().is_empty() {
                fail("constitution is empty");
            }
            let rt = open_rt(&cli.state);
            rt.append(Event::Init {
                scope: scope.clone(),
                memories_hash: fingerprint(&md),
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
        Cmd::Next { scope, seed } => {
            let rt = open_rt(&cli.state);
            let data = load_scope(&rt, &scope);
            let n = data.items.len() as u32;
            if n < 2 {
                fail("need at least 2 memories to deal a pair");
            }
            let voted = voted_set(&rt, &scope);
            let pendings = pending_list(&rt, &scope);
            let pending_pairs: HashSet<(u32, u32)> = pendings
                .iter()
                .map(|(_, p)| if p.a < p.b { (p.a, p.b) } else { (p.b, p.a) })
                .collect();
            // Overlap-shy: don't re-deal items from the most recent deal
            // still awaiting judgment, so the agent sees fresh meat first.
            let current: Option<(u32, u32)> = pendings
                .iter()
                .map(|(_, p)| (p.a, p.b))
                .max_by_key(|(a, b)| (*a as u64) << 32 | *b as u64)
                .map(|(a, b)| if a <= b { (a, b) } else { (b, a) });
            let dealt = match seed {
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
            let sc = scores(&rt, &scope, data.items.len());
            let mut ranked: Vec<(u32, f64)> =
                (0..data.items.len() as u32).map(|i| (i, sc[i as usize])).collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let items: Vec<_> = ranked
                .into_iter()
                .take(top)
                .enumerate()
                .map(|(r, (i, s))| {
                    serde_json::json!({
                        "rank": r + 1,
                        "id": mem_id(i),
                        "score": s,
                        "text": data.items[i as usize],
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::json!({
                    "scope": scope,
                    "votes": vote_count(&rt, &scope),
                    "items": items,
                })
            );
        }
        Cmd::Render { scope, top, output } => {
            let rt = open_rt(&cli.state);
            let data = load_scope(&rt, &scope);
            let sc = scores(&rt, &scope, data.items.len());
            let mut ranked: Vec<(u32, f64)> =
                (0..data.items.len() as u32).map(|i| (i, sc[i as usize])).collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let votes = vote_count(&rt, &scope);
            let mut md = format!(
                "# top memories — {scope} ({votes} votes)\n\n",
            );
            for (r, (i, s)) in ranked.into_iter().take(top).enumerate() {
                md.push_str(&format!("## {}. {} · score {s:.4}\n\n{}\n\n", r + 1, mem_id(i), data.items[i as usize]));
            }
            fs::write(&output, md).unwrap_or_else(|e| fail(format!("write {}: {e}", output.display())));
            println!(
                "{}",
                serde_json::json!({"ok": true, "output": output.display().to_string(), "votes": votes})
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
}
