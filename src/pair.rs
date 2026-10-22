//! Pair choosing, ported from scope_rank.rs's pool logic.
//!
//! scope_rank speaks ItemId + GroupState from a bigger body that doesn't
//! live in this crate. This is her heart, beating in plain u32: degree
//! tracking so no item gets starred, bridging-first when the graph is
//! disconnected, overlap deprioritization so the agent isn't re-dealt the
//! pair it just saw. Same keys, same order, same tests — smaller mouth.

use std::collections::{HashMap, HashSet};
use std::cmp::Reverse;

/// Canonical unordered pair: (lo, hi). Voted/pending sets hold these.
fn canon(a: u32, b: u32) -> (u32, u32) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Connected-component ids for the 0..n pool, joined only by voted pairs.
/// Never-voted items are singleton components. One id per index.
pub fn component_ids(n: u32, voted: &HashSet<(u32, u32)>) -> Vec<usize> {
    let n = n as usize;
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            if ra < rb {
                parent[rb] = ra;
            } else {
                parent[ra] = rb;
            }
        }
    }
    for &(a, b) in voted {
        let (a, b) = (a as usize, b as usize);
        if a < n && b < n && a != b {
            union(&mut parent, a, b);
        }
    }
    let mut root_to_id: HashMap<usize, usize> = HashMap::new();
    let mut out = vec![0usize; n];
    for i in 0..n {
        let r = find(&mut parent, i);
        let next = root_to_id.len();
        out[i] = *root_to_id.entry(r).or_insert(next);
    }
    out
}

/// Deal the next pair: bridging-first across components, else lowest-degree,
/// never voted, never already pending, never excluded. `current` is the pair
/// the agent just saw (deal or vote) — sharing its items costs overlap.
/// Returns None when every pair is voted or pending.
pub fn suggest_next_pair(
    n: u32,
    voted: &HashSet<(u32, u32)>,
    pending: &HashSet<(u32, u32)>,
    current: Option<(u32, u32)>,
    excluded: Option<&HashSet<(u32, u32)>>,
) -> Option<(u32, u32)> {
    let n_usize = n as usize;
    if n < 2 {
        return None;
    }
    let current = current.map(|(a, b)| canon(a, b));
    let is_excluded = |pair: &(u32, u32)| excluded.is_some_and(|s| s.contains(pair));
    let is_open = |a: u32, b: u32| {
        let p = canon(a, b);
        voted.contains(&p) || pending.contains(&p)
    };

    // Degree inside the pool: how many voted opponents each item already has.
    // Lex-first unvoted scan starred the earliest item until every one of
    // its pairs was judged; degree preference spreads the love instead.
    let mut degree = vec![0usize; n_usize];
    for &(a, b) in voted {
        let (a, b) = (a as usize, b as usize);
        if a < n_usize && b < n_usize && a != b {
            degree[a] += 1;
            degree[b] += 1;
        }
    }
    let overlap = |a: u32, b: u32| match &current {
        Some((x, y)) => u8::from(a == *x || a == *y) + u8::from(b == *x || b == *y),
        None => 0,
    };

    // When the pool subgraph is disconnected, a vote inside one component
    // can never connect it to the others. Prefer a bridging pair first: any
    // single cross-component vote merges two components. Among bridges,
    // attach to the largest components first (a singleton joins the main
    // ranking rather than pairing singletons together), then keep the
    // low-degree preference so one item is not starred.
    let comp = component_ids(n, voted);
    let num_components = comp.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    if num_components > 1 {
        let mut comp_size = vec![0usize; num_components];
        for &c in &comp {
            comp_size[c] += 1;
        }
        let mut best: Option<(u8, Reverse<usize>, usize, usize, u32, u32)> = None;
        for a in 0..n {
            for b in (a + 1)..n {
                if comp[a as usize] == comp[b as usize] {
                    continue;
                }
                let pair = canon(a, b);
                if current.as_ref() == Some(&pair) || is_excluded(&pair) {
                    continue;
                }
                if is_open(a, b) {
                    continue;
                }
                let size_sum = comp_size[comp[a as usize]] + comp_size[comp[b as usize]];
                let key = (
                    overlap(a, b),
                    Reverse(size_sum),
                    degree[a as usize].max(degree[b as usize]),
                    degree[a as usize] + degree[b as usize],
                    a,
                    b,
                );
                if best.as_ref().map_or(true, |best| key < *best) {
                    best = Some(key);
                }
            }
        }
        if let Some((_, _, _, _, a, b)) = best {
            return Some((a, b));
        }
    }

    let mut best: Option<(u8, usize, usize, u32, u32)> = None;
    for a in 0..n {
        for b in (a + 1)..n {
            let pair = canon(a, b);
            if current.as_ref() == Some(&pair) || is_excluded(&pair) {
                continue;
            }
            if is_open(a, b) {
                continue;
            }
            let key = (
                overlap(a, b),
                degree[a as usize].max(degree[b as usize]),
                degree[a as usize] + degree[b as usize],
                a,
                b,
            );
            if best.as_ref().map_or(true, |best| key < *best) {
                best = Some(key);
            }
        }
    }
    best.map(|(_, _, _, a, b)| (a, b))
}

/// Pick the best partner for a pinned item: bridging-first across components,
/// else lowest-degree. Never voted, never already pending. `None` when every
/// pair containing `a` is voted or pending.
/// The pinned side is explicit, so no `current` overlap penalty applies here.
pub fn suggest_partner_for(
    a: u32,
    n: u32,
    voted: &HashSet<(u32, u32)>,
    pending: &HashSet<(u32, u32)>,
) -> Option<u32> {
    if n < 2 || a >= n {
        return None;
    }
    let comp = component_ids(n, voted);
    let mut comp_size = vec![0usize; comp.iter().copied().max().map(|m| m + 1).unwrap_or(0)];
    for &c in &comp {
        comp_size[c] += 1;
    }
    let mut degree = vec![0usize; n as usize];
    for &(x, y) in voted {
        let (x, y) = (x as usize, y as usize);
        if x < n as usize && y < n as usize && x != y {
            degree[x] += 1;
            degree[y] += 1;
        }
    }
    let mut best: Option<(u8, Reverse<usize>, usize, u32)> = None;
    for b in 0..n {
        if b == a {
            continue;
        }
        if voted.contains(&canon(a, b)) || pending.contains(&canon(a, b)) {
            continue;
        }
        // Cross-component first (0 beats 1), then the biggest target
        // component, then the least-voted partner, then lowest index.
        let cross = u8::from(comp[a as usize] == comp[b as usize]);
        let key = (
            cross,
            Reverse(comp_size[comp[b as usize]]),
            degree[b as usize],
            b,
        );
        if best.as_ref().map_or(true, |best| key < *best) {
            best = Some(key);
        }
    }
    best.map(|(_, _, _, b)| b)
}
#[cfg(test)]
mod tests {
    use super::*;

    fn voted(pairs: &[(u32, u32)]) -> HashSet<(u32, u32)> {
        pairs.iter().map(|&(a, b)| canon(a, b)).collect()
    }

    #[test]
    fn skips_current_and_voted_pairs() {
        let v = voted(&[(0, 1)]);
        let next = suggest_next_pair(3, &v, &HashSet::new(), Some((0, 1)), None).unwrap();
        assert!(next.0 == 2 || next.1 == 2);
        assert_ne!(canon(next.0, next.1), (0, 1));
    }

    #[test]
    fn does_not_star_first_item() {
        // Item 0 has judged everyone; the next deal must leave it alone.
        let v = voted(&[(0, 1), (0, 2), (0, 3)]);
        let next = suggest_next_pair(4, &v, &HashSet::new(), Some((0, 1)), None).unwrap();
        assert_ne!(next.0, 0);
        assert_ne!(next.1, 0);
        // Landing with no current must not restar it either.
        let landing = suggest_next_pair(4, &v, &HashSet::new(), None, None).unwrap();
        assert_ne!(landing.0, 0);
        assert_ne!(landing.1, 0);
    }

    #[test]
    fn prefers_bridging_vote_when_disconnected() {
        // 3-node component plus 2-node component: degree-only logic ties the
        // open intra-component pair against every cross pair, and lex order
        // would pick the intra pair — which can never connect the graph.
        let v = voted(&[(0, 1), (1, 2), (3, 4)]);
        let next = suggest_next_pair(5, &v, &HashSet::new(), None, None).unwrap();
        let in_abc = |x: u32| x <= 2;
        let in_de = |x: u32| x >= 3;
        assert!(
            (in_abc(next.0) && in_de(next.1)) || (in_de(next.0) && in_abc(next.1)),
            "expected a bridging pair, got {next:?}"
        );
    }

    #[test]
    fn attaches_singleton_to_largest_component() {
        let v = voted(&[(0, 1), (1, 2)]);
        let next = suggest_next_pair(4, &v, &HashSet::new(), None, None).unwrap();
        assert!(
            next.0 == 3 || next.1 == 3,
            "expected the singleton attached, got {next:?}"
        );
    }

    #[test]
    fn pending_pairs_are_not_redealt() {
        let pending: HashSet<(u32, u32)> = [canon(0, 1)].into_iter().collect();
        let next = suggest_next_pair(3, &HashSet::new(), &pending, None, None).unwrap();
        assert_ne!(canon(next.0, next.1), (0, 1));
    }

    #[test]
    fn excluded_pairs_stay_out() {
        let excluded: HashSet<(u32, u32)> = [canon(0, 1)].into_iter().collect();
        let next = suggest_next_pair(3, &HashSet::new(), &HashSet::new(), None, Some(&excluded)).unwrap();
        assert_ne!(canon(next.0, next.1), (0, 1));
        assert!(next.0 == 2 || next.1 == 2);
    }

    #[test]
    fn done_when_everything_voted() {
        let v = voted(&[(0, 1), (0, 2), (1, 2)]);
        assert_eq!(suggest_next_pair(3, &v, &HashSet::new(), None, None), None);
    }

    #[test]
    fn overlap_avoids_current_items() {
        // Nothing voted: with current (0,1), the clean pair avoids both.
        let next = suggest_next_pair(4, &HashSet::new(), &HashSet::new(), Some((0, 1)), None).unwrap();
        assert_eq!(canon(next.0, next.1), (2, 3));
    }

    #[test]
    fn pinned_partner_prefers_bridge() {
        // a=0 sits in {0,1}; the best partner bridges to the big component.
        let v = voted(&[(0, 1), (2, 3), (3, 4)]);
        assert_eq!(suggest_partner_for(0, 5, &v, &HashSet::new()), Some(2));
    }

    #[test]
    fn pinned_partner_avoids_voted_and_pending() {
        let v = voted(&[(0, 1)]);
        let pending: HashSet<(u32, u32)> = [canon(0, 2)].into_iter().collect();
        assert_eq!(suggest_partner_for(0, 3, &v, &pending), None);
        assert_eq!(suggest_partner_for(0, 4, &v, &pending), Some(3));
    }
}
