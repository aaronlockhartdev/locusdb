extern crate test;

use std::default::Default;
use std::ptr::NonNull;
use std::cmp::{PartialOrd, Ordering};

use rand::random;

use bumpalo::Bump;

use super::{MemTable, Collision, NotFound};

// SkipList variant outlined in https://doi.org/10.1016/j.jda.2011.12.017
//
// SAFETY: Makes no guarantees about thread safety, as its meant to be used
// in a thread-per-core environment
pub struct SkipLift<T, const H: usize> {
    // Due to bump allocation, it doesn't make sense to remove sentinel
    // nodes (because their memory won't be freed from the global allocator),
    // so we store them as a fixed size array of max_height
    sentinels: [Link<T>; H],
    alloc: Bump
}

type Link<T> = Option<NonNull<Node<T>>>;

struct Node<T> {
    // T will often be a (K, *mut V) or (K, V) pair. 
    //
    // In the former case, V should have its own allocator to help with 
    // traversal locality. On consumption (usually as iter), V's ptr will 
    // have to be dropped manually.
    // 
    // The latter case should be used when V is small enough not to warrant
    // the storage of a pointer.
    data: T,
    // ptr[0] is refering to the nodes before and after at self.height,
    // while ptr[1] is refering to the nodes before and after on the level
    // below
    prev: [Link<T>; 2],
    next: [Link<T>; 2],
    height: u8
}

impl<T, const H: usize> SkipLift<T, H> where
    T: Copy + PartialOrd {
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
        // Assuming H < 128, sample a uniform distribution, then mask
        // to limit max_height and FFZ (find first zero)
        let bits = random::<u128>() & ((0x1u128 << (H - 2)) - 1);
        bits.trailing_ones() as usize + 1
    }

    // Implementation of general search/insert/update function
    pub fn upsert_if<F>(&mut self, data: T, mut cond: F) where 
        F: FnMut(Option<T>) -> bool {

        // Initialize initial node and height
        let mut h = H - 1;
        let mut n: Link<T> = None;

        // Allocate new node to stack
        let mut node = Node::<T> {
            data,
            prev: [None; 2],
            next: [None; 2],
            height: Self::gen_height() as u8
        };

        // idx is 0 or 1, keeps track of node.height vs node.height - 1
        let mut idx = 0;

        // Search for value
        while h > 0 {
            while let Some(_n) = n {
                let _n = unsafe { _n.as_ref() };

                if (_n.height as usize) == h { break; }
                else { n = _n.prev[1]; }
            } 

            h -= 1;

            while let Some(mut next) = n.map_or(
                self.sentinels[h], 
                |_n| {
                    let _n_ref = unsafe { _n.as_ref() };
                    _n_ref.next[_n_ref.height as usize - h]
                }
            ) {
                let next_ref = unsafe { next.as_mut() };

                match next_ref.data.partial_cmp(&data).unwrap() {
                    Ordering::Less => { 
                        // Need to keep moving right
                        n = Some(next);
                    },
                    Ordering::Equal => {
                        // Value found, update if cond(value)
                        if cond(Some(next_ref.data)) {
                            next_ref.data = data;
                        }
                        return;
                    },
                    Ordering::Greater => { break; }
                }
            }
            
            // Keep track of current, next nodes for later insertion
            // if we're at node.height or node.height - 1
            if idx < 2 && h == (node.height as usize) - idx {
                node.prev[idx] = n;
                node.next[idx] = n.map_or(
                    self.sentinels[h], 
                    |_n| {
                        let _n_ref = unsafe { _n.as_ref() };
                        _n_ref.next[_n_ref.height as usize - h]
                    }
                );

                idx += 1;
            }
        }

        // If value not found, insert only if cond(None)
        if !cond(None) {
            return;
        }
        
        // Allocate space in bump allocator
        let node_ptr = unsafe { 
            NonNull::new_unchecked(
                self.alloc.alloc(node)
            )
        };

        let node_ref = unsafe {
            node_ptr.as_ref()
        };

        // Insert node between previously found nodes
        for idx in 0..=1 {
            if let Some(mut prev) = node_ref.prev[idx] {
                let prev_ref = unsafe { prev.as_mut() };
                let prev_idx = ((prev_ref.height + idx as u8) - node_ref.height) as usize;
                prev_ref.next[prev_idx] = Some(node_ptr);
            } else {
                self.sentinels[node_ref.height as usize - idx] = Some(node_ptr);
            }

            if let Some(mut next) = node_ref.next[idx] {
                let next_ref = unsafe { next.as_mut() };
                let next_idx = ((next_ref.height + idx as u8) - node_ref.height) as usize;

                next_ref.prev[next_idx] = Some(node_ptr);
            }
        }
    }

}

// Implement MemTable for SkipLift
#[derive(Copy, Clone)]
struct KeyVal<K, V>(K, V)
    where K: Copy, V: Copy;

impl<K, V> PartialEq for KeyVal<K, V> where 
    K: PartialEq + Copy, V: Copy {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<K, V> PartialOrd for KeyVal<K, V> where
    K: PartialOrd + Copy, V: Copy {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<K, V, const H: usize> MemTable<K, V> for SkipLift<KeyVal<K, V>, H> where 
    K: PartialOrd + Copy, V: Default + Copy {

    fn new(size: Option<usize>) -> Self { 
        let lift = Self::new();
        lift.alloc.set_allocation_limit(size);
        lift
    }

    fn create(&mut self, key: K, val: V) -> Result<(), Collision> {
        let mut res = Default::default();
        SkipLift::upsert_if(
            self, 
            KeyVal(key, val),
            |d| {
                res = d.is_none();
                d.is_none()
            }
        );

        if res { Ok(()) } else { Err(Collision) }
    }

    fn read(&mut self, key: K) -> Result<V, NotFound> {
        let mut val = Default::default();
        SkipLift::upsert_if(
            self, 
            KeyVal(key, Default::default()), 
            |d| {
                val = d.map(|v| v.1);
                false
            }
        );

        val.ok_or(NotFound)
    }
    
    fn update(&mut self, key: K, new_val: V) -> Result<V, NotFound> {
        let mut val = Default::default();
        SkipLift::upsert_if(
            self, 
            KeyVal(key, new_val), 
            |d| {
                val = d.map(|v| v.1);
                d.is_some()
            }
        );
        
        val.ok_or(NotFound)
    }
}

#[cfg(test)]
mod tests {
    use rand::random;
    use std::{collections::BTreeMap, time};

    use super::{KeyVal, SkipLift};
    use crate::store::memtable::{self, MemTable};


    #[test]
    fn basic_small() {
        let mut lift = SkipLift::<i64, 16>::new();
            
        for _ in 0..1_000 {
            lift.upsert_if(1, |x| match x {
                Some(_) => true,
                None => true
            });
        }
    }

    #[test]
    fn memtable_small() {
        let mut lift = <
            SkipLift::<KeyVal<usize, usize>, 32> as 
            memtable::MemTable<usize, usize>
            >::new(None);

        let key: usize = random();

        lift.create(key, random()).unwrap();

        for i in 0..1_000 {
            assert!(lift.create(key, i).is_err());

            let _ = lift.create(random(), i);
        }

        lift.update(key, 1).unwrap();

        assert!(lift.read(key).unwrap() == 1);

        let start = time::Instant::now();

        let iters = 1_000_000;

        for _ in 0..iters {
            let _ = lift.read(key);
        }

        println!("{:.2?}", start.elapsed() / iters);
    }

    use test::Bencher;

    
    #[bench]
    fn memtable_read(b: &mut Bencher) {
        let mut lift = <
            SkipLift::<KeyVal<usize, usize>, 32> as 
            memtable::MemTable<usize, usize>
            >::new(None);


        let keys: Vec<usize> = (0..1_000_000).map(|_| random()).collect();
        let k: usize = random();


        for k in keys {
            let _ = lift.create(k, 0);
        }

        b.iter(|| {
            let _ = lift.read(k);
        });

    }

    #[bench]
    fn btree_read(b: &mut Bencher) {
        let mut tree: BTreeMap<usize, usize> = std::collections::BTreeMap::new();

        let keys: Vec<usize> = (0..1_000_000).map(|_| random()).collect();
        let k: usize = random();


        for k in keys {
            tree.insert(k, 0);
        }

        b.iter(|| {
            let _ = tree.get(&k);
        });

    }
}
