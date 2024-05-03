extern crate test;

use std::mem::{replace, swap, take, MaybeUninit};
use std::ptr::NonNull;
use std::cmp::Ord;

use rand::random;

use bumpalo::Bump;

// SkipList variant outlined in https://doi.org/10.1016/j.jda.2011.12.017
//
// SAFETY: Makes no guarantees about thread safety, as its meant to be used
// in a thread-per-core database environment
pub struct SkipLift<K, V, const C: usize, const H: usize> where
    K: PartialOrd {
    // Due to bump allocation, removing and re-allocating sentinel nodes will
    // cause memory bloat (because their memory won't be freed from the global 
    // allocator) so we store them as a fixed size array of max_height
    sentinels: [Link<K, V, C>; H],
    // Bump allocator for fast allocations and cache locality, because 
    // MemTables are append-only anyways
    alloc: Bump,
}

type Link<K, V, const C: usize> = Option<NonNull<Node<K, V, C>>>;

struct Node<K, V, const C: usize> {
    // Cache optimization unrolling similar to b-trees
    len: u8,
    keys: [K; C],
    vals: [V; C],
    // ptr[0] is referring to the nodes before and after at self.height,
    // while ptr[1] is refering to the nodes before and after on the level
    // below
    prev: [Link<K, V, C>; 2],
    next: [Link<K, V, C>; 2],
    // Height should be log_p(n), so in most cases will be <= 255
    height: u8,
}


impl<K, V, const C: usize> Node<K, V, C> where
    K: Ord + std::fmt::Debug {
    fn search(&self, key: &K) -> Option<&V> {
        // Simple linear search over chunk
        //
        // TODO: Optimize with different search method

        for i in 0..self.len as usize {
            if key == &self.keys[i] {
                return Some(&self.vals[i]);
            }
        }
        None
    }

    fn insert(&mut self, key: K, val: V) -> Result<Option<V>, (K, V)> {

        for i in 0..self.len() {
            // Iterate over elements
            if &self.keys[i] == &key {
                // If we found an element, return Ok(Some)
                return Ok(Some(replace(&mut self.vals[i], val)));
            } else if &self.keys[i] > &key && !self.is_full() {
                // If we passed the element and we have room,
                // shift elements right and insert
                self.len += 1;

                let len = self.len();

                (&mut self.keys[i..len]).rotate_right(1);
                (&mut self.vals[i..len]).rotate_right(1);

                self.keys[i] = key;
                self.vals[i] = val;

                return Ok(None);
            }
        }

        if self.is_full() {
            // If we're full, error and pass back ownership
            return Err((key, val));
        }

        // If all elements are < key, append to the end
        self.keys[self.len()] = key;
        self.vals[self.len()] = val;
        self.len += 1;
        Ok(None)
        
    }

    fn upper_half(&mut self) -> ([K; C], [V; C], usize) {
        let mut keys: [K; C] = unsafe { MaybeUninit::uninit().assume_init() };
        let mut vals: [V; C] = unsafe { MaybeUninit::uninit().assume_init() };

        let offset = self.len() / 2;
        let new_len = self.len() - offset;

        for i in 0..new_len {
            swap(&mut keys[i], &mut self.keys[i + offset]);
            swap(&mut vals[i], &mut self.vals[i + offset]);
        }
        self.len = offset as u8;

        (keys, vals, new_len)
    }

    fn lower_half(&mut self) -> ([K; C], [V; C], usize) {
        
        let (mut keys, mut vals, mut len) = self.upper_half();

        let new_len = self.len();

        swap(&mut self.keys, &mut keys);
        swap(&mut self.vals, &mut vals);
        self.len = len as u8;

        (keys, vals, new_len)

    }

    fn height(&self) -> usize {
        self.height as usize
    }

    fn len(&self) -> usize {
        self.len as usize
    }

    fn min(&self) -> &K {
        &self.keys[0]
    }

    fn max(&self) -> &K {
        &self.keys[self.len() - 1]
    }

    fn is_full(&self) -> bool {
        self.len() == C
    }
}

impl<K, V, const C: usize, const H: usize> SkipLift<K, V, C, H> where
    K: Ord + std::fmt::Debug, V: std::fmt::Debug {
    // SAFETY: Panics if OOM
    fn new() -> Self {
        Self {
            sentinels: [None; H],
            alloc: Bump::new()
        }
    }

    fn gen_height() -> usize {
        // Efficiently compute height from Geometric(0.5) distribution, 
        // credit to https://ticki.github.io/blog/skip-lists-done-right/
        //
        // Assuming H < 64, sample a uniform distribution, then mask
        // to limit max_height and FFZ (find first zero)
        let bits = random::<u64>() & ((0x1u64 << (H - 2)) - 1);
        bits.trailing_ones() as usize + 1
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        let mut h = H;
        let mut cn: Link<K, V, C> = None;

        while h > 0 {
            while let Some(n) = cn {
                let n_ref = unsafe { n.as_ref() };

                if n_ref.height() == h { break; }
                else { cn = n_ref.prev[1]; }
            }

            h -= 1;

            let mut nn = cn.map_or(self.sentinels[h], |n| {
                let n_ref = unsafe { n.as_ref() };
                n_ref.next[n_ref.height() - h]
            });

            while let Some(n) = nn {
                let n_ref = unsafe { n.as_ref() }; 

                if n_ref.min() > &key {
                    break;
                } else if n_ref.max() <= &key {
                    if let Some(v) = n_ref.search(key) {
                        return Some(v);
                    }
                    break;
                } else {
                    cn = nn;
                    nn = n_ref.next[n_ref.height() - h];
                }
            }
        }

        None
    }

    pub fn insert(&mut self, mut key: K, mut val: V) -> Option<V> {
        let mut h = H;
        let mut cn: Link<K, V, C> = None;

        // Pre-generate height in case we need to add a node
        // (we can store previous and next pointers)
        let new_h = Self::gen_height();

        let mut new_pn = [None; 2];
        let mut new_nn = [None; 2];

        // Best nodes to insert key into
        let mut best: [Link<K, V, C>; 2] = [None; 2];

        while h > 0 {
            // While we're not at a sentinel and we can't go down,
            // go left
            while let Some(n) = cn {
                let n_ref = unsafe { n.as_ref() };
                if n_ref.height() == h { 
                    break; 
                }
                else { 
                    cn = n_ref.prev[1]; 
                }
            }

            // Go down
            h -= 1;

            // Initialize next node
            let mut nn = cn.map_or(self.sentinels[h], |n| {
                let n_ref = unsafe { n.as_ref() };
                n_ref.next[1];
                Some(n)
            });

            // While we can go right, do so until we overshoot
            while let Some(mut n) = nn {
                let n_ref = unsafe { n.as_mut() };

                if n_ref.min() > &key { break; }

                cn = nn;
                nn = n_ref.next[n_ref.height() - h];
            }

            if best[0].is_none() || best[0].is_some_and(|n| {
                let n_ref = unsafe { n.as_ref() };
                cn.is_some_and(|_n| {
                    let _n_ref = unsafe { _n.as_ref() }; 
                    let res = _n_ref.max() > n_ref.max();
                    res
                })
            }){
                best[0] = cn;
            }

            if best[0].is_some_and(|n| {
                let n_ref = unsafe { n.as_ref() };

                n_ref.max() > &key
            }) {
                best[1] = None;
            } else {
                if best[1].is_none() || best[1].is_some_and(|n| {
                    let n_ref = unsafe { n.as_ref() };
                    nn.is_some_and(|_n| {
                        let _n_ref = unsafe { _n.as_ref() }; 
                        _n_ref.min() < n_ref.min()
                    })

                }){
                    best[1] = nn;
                }
            }

            if new_h >= h && new_h - h < 2 {
                new_nn[new_h - h] = nn;
                new_pn[new_h - h] = cn;
            }
        }

        for bn in best {
            if let Some(mut n) = bn {
                let n_ref = unsafe { n.as_mut() };

                


                match n_ref.insert(key, val) {
                    Ok(old_v) => { 
                        return old_v; 
                    } Err((k, v)) => { 
                        key = k;
                        val = v; 
                    }
                }
                if n_ref.max() >= &key {
                    break;
                }
            }
        } 
        
        
        // Need to create new node
        let new_n = unsafe { NonNull::new_unchecked(self.alloc.alloc_with(|| {
            if let Some(mut n) = best[0] {
                let n_ref = n.as_mut();
                let (keys, vals, len) = n_ref.upper_half();
                let mut n = Node {
                    height: new_h as u8,
                    keys,
                    vals,
                    len: len as u8,
                    next: new_nn,
                    prev: new_pn
                };
                assert!(n_ref.max() < n.min());
                if n_ref.max() >= &key {
                    n_ref.insert(key, val);
                } else {
                    n.insert(key, val);
                }
                n
            } else if let Some(mut n) = best[1] {
                let n_ref = n.as_mut();
                let (keys, vals, len) = n_ref.lower_half();
                let mut n = Node {
                    height: new_h as u8,
                    keys,
                    vals,
                    len: len as u8,
                    next: new_nn,
                    prev: new_pn
                };
                assert!(n.len() > 0);
                assert!(n.max() < n_ref.min());
                if n.max() >= &key {
                    n.insert(key, val);
                } else {
                    n_ref.insert(key, val);
                }
                n
            } else { 
                let (mut keys, mut vals): ([K; C], [V; C]) = {
                    MaybeUninit::zeroed().assume_init()
                };
                keys[0] = key;
                vals[0] = val;
                let mut n = Node {
                    height: new_h as u8,
                    keys,
                    vals,
                    len: 1u8,
                    next: new_nn,
                    prev: new_pn
                };
                n
            }
            
        }))};

        unsafe {
            let new_n_ref = new_n.as_ref();

            for i in 0..=1 {
                if let Some(mut nn) = new_n_ref.next[i] {
                    let nn_ref = nn.as_mut();
                    nn_ref.prev[
                        nn_ref.height() + i - new_n_ref.height()
                    ] = Some(new_n);
                }

                if let Some(mut pn) = new_n_ref.prev[i] {
                    let pn_ref = pn.as_mut();
                    pn_ref.next[
                        pn_ref.height() + i - new_n_ref.height()
                    ] = Some(new_n);
                    
                    
                } else {
                    self.sentinels[new_h - i] = Some(new_n);
                }
            }
        }

        None
    }

}

mod tests {
    use rand::random;
    use std::{collections::BTreeMap, time};

    use super::SkipLift;

    #[test]
    fn memtable_small() {
        let mut lift: SkipLift<u8, u8, 4, 6> = SkipLift::new();

        const ITERS: usize = 10;
        let key: u8 = random();

        lift.insert(key, random());

        for i in 0..ITERS as u8{
            let _ = lift.insert(random(), i);
            assert!(lift.insert(key, i).is_some());
        }

        assert!(lift.insert(key, random()).unwrap() == ITERS as u8 - 1);

        let start = time::Instant::now();

        let iters = 1_000_000;

        for _ in 0..iters {
            let _ = lift.get(&key);
        }

        println!("{:.2?}", start.elapsed() / iters);
    }

    use test::Bencher;

    #[bench]
    fn memtable_insert(b: &mut Bencher) {
        let mut lift: SkipLift<u8, u8, 8, 10> = SkipLift::new();

        let keys: Vec<u8> = (0..1_000).map(|_| random()).collect();

        b.iter(|| {
            for k in &keys {
                let _ = lift.insert(*k, 0);
            }
        })
    }

    #[bench]
    fn btree_insert(b: &mut Bencher) {
        let mut tree: BTreeMap<u8, u8> = std::collections::BTreeMap::new();

        let keys: Vec<u8> = (0..1_000).map(|_| random()).collect();

        
        b.iter(|| {
            for k in &keys {
                tree.insert(*k, 0);
            }
        });

    }

    #[bench]
    fn memtable_read(b: &mut Bencher) {
        let mut lift: SkipLift<u8, u8, 8, 10> = SkipLift::new();

        let keys: Vec<u8> = (0..1_000).map(|_| random()).collect();


        for k in &keys {
            let _ = lift.insert(*k, 0);
        }

        b.iter(|| {
            for k in &keys {
                let _ = lift.get(k);
            }
        });

    }

    #[bench]
    fn btree_read(b: &mut Bencher) {
        let mut tree: BTreeMap<u8, u8> = std::collections::BTreeMap::new();

        let keys: Vec<u8> = (0..1_000).map(|_| random()).collect();

        for k in &keys {
            tree.insert(*k, 0);
        }

        b.iter(|| {
            for k in &keys {
                let _ = tree.get(k);
            }
        });

    }
}
