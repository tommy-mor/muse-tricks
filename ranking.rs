use std::collections::{BTreeSet, HashMap, HashSet};

use crate::path_types::ItemId;
use crate::reducer::{canonical_pair_ids, ScopeVotes};

#[derive(Debug, Clone)]
pub struct RankedItem {
    pub item: ItemId,
    pub score: f64,
}

pub const MAX_ITERS: usize = 10_000;
pub const TOL: f64 = 1e-8;

pub fn item_index(scope: &ScopeVotes) -> (HashMap<ItemId, usize>, Vec<ItemId>) {
    let mut item_strs: BTreeSet<String> = BTreeSet::new();
    for vote in scope.uuid_votes.values() {
        item_strs.insert(vote.a.as_str().to_string());
        item_strs.insert(vote.b.as_str().to_string());
    }
    let mut idx_to_item: Vec<ItemId> = Vec::with_capacity(item_strs.len());
    let mut item_to_idx: HashMap<ItemId, usize> = HashMap::with_capacity(item_strs.len());
    for s in item_strs {
        let id = ItemId::from_storage(&s).unwrap_or_else(|| ItemId::opaque(&s));
        let idx = idx_to_item.len();
        item_to_idx.insert(id.clone(), idx);
        idx_to_item.push(id);
    }
    (item_to_idx, idx_to_item)
}

pub fn edges_from_scope(scope: &ScopeVotes) -> HashMap<(usize, usize), f64> {
    let (item_to_idx, _) = item_index(scope);
    let mut edges: HashMap<(usize, usize), f64> = HashMap::new();
    for vote in scope.uuid_votes.values() {
        let Some(&ai) = item_to_idx.get(&vote.a) else {
            continue;
        };
        let Some(&bi) = item_to_idx.get(&vote.b) else {
            continue;
        };
        let w_a = vote.ratio_left as f64 * vote.trust_weight;
        let w_b = vote.ratio_right as f64 * vote.trust_weight;
        if w_a > 0.0 {
            *edges.entry((bi, ai)).or_insert(0.0) += w_a;
        }
        if w_b > 0.0 {
            *edges.entry((ai, bi)).or_insert(0.0) += w_b;
        }
    }
    edges
}

pub fn edge_weight_sum(scope: &ScopeVotes) -> f64 {
    edges_from_scope(scope).values().sum()
}

pub fn voted_pair_indices(scope: &ScopeVotes) -> HashSet<(usize, usize)> {
    let (item_to_idx, _) = item_index(scope);
    let mut pairs = HashSet::new();
    for vote in scope.uuid_votes.values() {
        let Some(&ai) = item_to_idx.get(&vote.a) else {
            continue;
        };
        let Some(&bi) = item_to_idx.get(&vote.b) else {
            continue;
        };
        let (i, j) = if ai < bi { (ai, bi) } else { (bi, ai) };
        pairs.insert((i, j));
    }
    pairs
}

pub fn pair_is_voted(scope: &ScopeVotes, a: &ItemId, b: &ItemId) -> bool {
    let (lo, hi) = canonical_pair_ids(a, b);
    scope
        .uuid_votes
        .keys()
        .any(|(_, l, h)| l == &lo && h == &hi)
}

pub fn connected_components_from_voted_pairs(
    n: usize,
    voted_pairs: impl Iterator<Item = (usize, usize)>,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (a, b) in voted_pairs {
        if a >= n || b >= n || a == b {
            continue;
        }
        adj[a].push(b);
        adj[b].push(a);
    }

    let mut isolates: Vec<usize> = (0..n).filter(|&i| adj[i].is_empty()).collect();
    isolates.sort();

    let mut seen = vec![false; n];
    for &i in &isolates {
        seen[i] = true;
    }

    let mut comps: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        if seen[i] {
            continue;
        }
        let mut stack = vec![i];
        seen[i] = true;
        let mut comp: Vec<usize> = Vec::new();
        while let Some(x) = stack.pop() {
            comp.push(x);
            for &y in &adj[x] {
                if !seen[y] {
                    seen[y] = true;
                    stack.push(y);
                }
            }
        }
        comp.sort();
        comps.push(comp);
    }

    (comps, isolates)
}

pub fn scope_components(scope: &ScopeVotes) -> (Vec<Vec<usize>>, Vec<usize>, Vec<ItemId>) {
    let (_, idx_to_item) = item_index(scope);
    let n = idx_to_item.len();
    let pairs = voted_pair_indices(scope);
    let (comps, isolates) = connected_components_from_voted_pairs(n, pairs.into_iter());
    (comps, isolates, idx_to_item)
}

pub fn ranked_items(scope: &ScopeVotes) -> Vec<RankedItem> {
    let (_, idx_to_item) = item_index(scope);
    let n = idx_to_item.len();
    let edges = edges_from_scope(scope);
    let scores = compute_scores_from_edges(n, edges.into_iter(), MAX_ITERS, TOL);

    let mut items: Vec<RankedItem> = idx_to_item
        .into_iter()
        .enumerate()
        .map(|(i, item)| RankedItem {
            item,
            score: *scores.get(i).unwrap_or(&0.0),
        })
        .collect();

    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    items
}

pub fn top_bottom(scope: &ScopeVotes, k: usize) -> (Vec<RankedItem>, Vec<RankedItem>) {
    let items = ranked_items(scope);
    if k == 0 || items.len() <= 2 * k {
        return (items, Vec::new());
    }
    let top = items[..k].to_vec();
    let bottom = items[items.len() - k..].to_vec();
    (top, bottom)
}

pub fn compute_scores_from_edges(
    n: usize,
    edges: impl Iterator<Item = ((usize, usize), f64)>,
    max_iters: usize,
    tol: f64,
) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }

    let mut raw: HashMap<(usize, usize), f64> = HashMap::new();
    for ((src, dst), w) in edges {
        if src >= n || dst >= n || w <= 0.0 {
            continue;
        }
        *raw.entry((src, dst)).or_insert(0.0) += w;
    }

    let keys: Vec<(usize, usize)> = raw.keys().copied().collect();
    let mut normalized: HashMap<(usize, usize), f64> = HashMap::new();
    for (i, j) in keys {
        if normalized.contains_key(&(i, j)) {
            continue;
        }
        let w_ij = *raw.get(&(i, j)).unwrap_or(&0.0);
        let w_ji = *raw.get(&(j, i)).unwrap_or(&0.0);
        let total = w_ij + w_ji;
        if total <= 0.0 {
            continue;
        }
        normalized.insert((i, j), w_ij / total);
        if w_ji > 0.0 {
            normalized.insert((j, i), w_ji / total);
        }
    }

    let mut out_edges: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    for ((src, dst), w) in &normalized {
        out_edges[*src].push((*dst, *w));
        neighbors[*src].insert(*dst);
        neighbors[*dst].insert(*src);
    }

    let weight_sum: Vec<f64> = out_edges
        .iter()
        .map(|es| es.iter().map(|(_, w)| *w).sum())
        .collect();
    let d_max = neighbors.iter().map(|s| s.len()).max().unwrap_or(0);
    if d_max == 0 {
        return vec![1.0 / n as f64; n];
    }
    let d_max_f = d_max as f64;

    let mut scores = vec![1.0 / n as f64; n];
    let mut next = vec![0.0f64; n];

    for _ in 0..max_iters {
        next.fill(0.0);
        for i in 0..n {
            let stay_prob = (d_max_f - weight_sum[i]) / d_max_f;
            next[i] += scores[i] * stay_prob;

            if out_edges[i].is_empty() {
                continue;
            }
            for &(dst, w) in &out_edges[i] {
                next[dst] += scores[i] * (w / d_max_f);
            }
        }

        let diff: f64 = scores
            .iter()
            .zip(next.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        scores.clone_from_slice(&next);
        if diff < tol {
            break;
        }
    }

    let sum: f64 = scores.iter().sum();
    if sum.is_finite() && sum > 0.0 {
        for s in &mut scores {
            *s /= sum;
        }
    }
    scores
}

pub fn ranked_items_subset(
    scope: &ScopeVotes,
    idxs: &[usize],
    max_iters: usize,
    tol: f64,
) -> Vec<RankedItem> {
    if idxs.is_empty() {
        return vec![];
    }

    let (_, idx_to_item) = item_index(scope);
    let edges = edges_from_scope(scope);

    let mut map: HashMap<usize, usize> = HashMap::with_capacity(idxs.len());
    for (j, &i) in idxs.iter().enumerate() {
        map.insert(i, j);
    }

    let edges_iter = edges.into_iter().filter_map(|((src, dst), w)| {
        let s = *map.get(&src)?;
        let d = *map.get(&dst)?;
        Some(((s, d), w))
    });

    let scores = compute_scores_from_edges(idxs.len(), edges_iter, max_iters, tol);

    let mut items: Vec<RankedItem> = idxs
        .iter()
        .enumerate()
        .filter_map(|(j, &orig)| {
            let item = idx_to_item.get(orig)?.clone();
            Some(RankedItem {
                item,
                score: *scores.get(j).unwrap_or(&0.0),
            })
        })
        .collect();

    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    items
}

pub fn group_summary_scores(scope: &ScopeVotes) -> HashMap<ItemId, f64> {
    ranked_items(scope)
        .into_iter()
        .map(|r| (r.item, r.score))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{DEFAULT_PSEUDONYM, TEST_ACTOR_UUID};
    use crate::reducer::VoteData;

    fn mk_scope() -> ScopeVotes {
        ScopeVotes::default()
    }

    fn vote(ts: i64, a: &str, b: &str, l: i32, r: i32) -> VoteData {
        VoteData::from_event(ts, a, b, l, r, DEFAULT_PSEUDONYM.to_string(), 1.0).unwrap()
    }

    fn apply(scope: &mut ScopeVotes, v: VoteData) {
        scope.apply_vote(v, TEST_ACTOR_UUID);
    }

    #[test]
    fn star_topology_winner_at_top_via_subset() {
        let mut scope = mk_scope();
        apply(&mut scope, vote(1, "zebra", "alpha", 2, 1));
        apply(&mut scope, vote(2, "zebra", "beta", 2, 1));

        let (_, idx_to_item) = item_index(&scope);
        let mut items: Vec<(usize, String)> = idx_to_item
            .iter()
            .enumerate()
            .map(|(i, it)| (i, it.as_str().to_string()))
            .collect();
        items.sort_by(|a, b| a.1.cmp(&b.1));
        let idxs: Vec<usize> = items.iter().map(|(i, _)| *i).collect();

        let ranked = ranked_items_subset(&scope, &idxs, 10000, 1e-8);
        assert_eq!(ranked[0].item.as_str(), "zebra");
    }

    #[test]
    fn top_bottom_splits_ends_without_overlap() {
        let mut scope = mk_scope();
        for (hi, lo) in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")] {
            apply(&mut scope, vote(1, hi, lo, 2, 1));
        }
        let (top, bottom) = top_bottom(&scope, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(bottom.len(), 2);
        for t in &top {
            assert!(bottom.iter().all(|b| b.item != t.item));
        }
        assert!(top[0].score >= bottom[bottom.len() - 1].score);
    }

    #[test]
    fn top_bottom_small_group_has_empty_bottom() {
        let mut scope = mk_scope();
        apply(&mut scope, vote(1, "a", "b", 2, 1));
        let (top, bottom) = top_bottom(&scope, 5);
        assert_eq!(top.len(), 2);
        assert!(bottom.is_empty());
    }

    #[test]
    fn connected_components_split_disconnected_pairs() {
        let mut scope = mk_scope();
        apply(&mut scope, vote(1, "a", "b", 3, 1));
        apply(&mut scope, vote(2, "c", "d", 3, 1));

        let (_, idx_to_item) = item_index(&scope);
        let (mut comps, isolates, _) = scope_components(&scope);
        assert!(isolates.is_empty());
        comps.sort_by_key(|c| {
            c.iter()
                .map(|&i| idx_to_item[i].clone())
                .collect::<Vec<_>>()
        });
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn twenty_five_random_votes_perfect_ratios_sort_alphabet() {
        use rand::seq::SliceRandom;

        const N: usize = 26;
        let letters: Vec<char> = (0..N).map(|i| char::from(b'a' + i as u8)).collect();

        let mut rng = rand::thread_rng();
        let mut perm: Vec<usize> = (0..N).collect();
        perm.shuffle(&mut rng);

        let mut scope = mk_scope();
        for k in 1..N {
            let i = *perm[..k].choose(&mut rng).unwrap();
            let j = perm[k];
            let (a, b) = (letters[i], letters[j]);
            apply(
                &mut scope,
                vote(
                    k as i64,
                    &a.to_string(),
                    &b.to_string(),
                    (i + 1) as i32,
                    (j + 1) as i32,
                ),
            );
        }

        let ranked = ranked_items(&scope);
        assert_eq!(ranked.len(), N);
        for (rank, item) in ranked.iter().enumerate() {
            let expected = char::from(b'a' + (N - 1 - rank) as u8);
            assert_eq!(item.item.as_str(), expected.to_string());
        }
    }

    #[test]
    fn subset_ranking_ranks_within_component_only() {
        let mut scope = mk_scope();
        apply(&mut scope, vote(1, "a", "b", 3, 1));
        apply(&mut scope, vote(2, "c", "d", 1, 4));

        let (comps, _, _) = scope_components(&scope);
        assert_eq!(comps.len(), 2);

        for comp in comps {
            let ranked = ranked_items_subset(&scope, &comp, 10000, 1e-8);
            assert_eq!(ranked.len(), 2);
            let names = ranked.iter().map(|r| r.item.as_str()).collect::<Vec<_>>();
            if names.contains(&"a") {
                assert_eq!(names[0], "a");
            } else {
                assert_eq!(names[0], "d");
            }
        }
    }
}
