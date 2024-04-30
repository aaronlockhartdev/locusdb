use core::panic;
use std::ptr::NonNull;
use std::cmp::{PartialOrd, Ordering};

use rand::random;

use bumpalo::Bump;

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
    next: [Link<T>; 2],
    prev: [Link<T>; 2],
    height: u8
}

struct Cursor<'a, T, const H: usize> {
    lift: &'a SkipLift<T, H>,
    // height is None iff we are at the header
    height: Option<usize>,
    // cur is None iff we are at a sentinel node
    cur: Link<T>
}

impl<T: PartialOrd + Copy, const H: usize> SkipLift<T, H> {
    // SAFETY: Panics if OOM
    pub fn new() -> Self {
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

    pub fn upsert_if<F>(&mut self, data: T, cond: F) where 
        F: Fn(Option<T>) -> bool {

        let mut h = H - 1;
        let mut n: Link<T> = None;

        let mut node = Node::<T> {
            data,
            prev: [None; 2],
            next: [None; 2],
            height: Self::gen_height() as u8
        };

        let mut idx = 0;

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
                        n = Some(next);
                    },
                    Ordering::Equal => {
                        if cond(Some(next_ref.data)) {
                            next_ref.data = data;
                        }
                        return;
                    },
                    Ordering::Greater => { break; }
                }
            }
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

        let node_ptr = unsafe { 
            NonNull::new_unchecked(
                self.alloc.alloc(node)
            )
        };

        let node_ref = unsafe {
            node_ptr.as_ref()
        };

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

    pub fn insert(&mut self, data: T) {
        let mut cur = Cursor::new(self);

        let height = Self::gen_height();
        
        let mut prev: [Link<T>; 2] = [None; 2];
        let mut next: [Link<T>; 2] = [None; 2];

        let mut h = height as i16;

        while {
            (h + 1) as usize >= height && 
            {
                cur.at_height(h as usize) || 
                cur.above_height(h as usize)
            }
        }{
            while {
                cur.peek_down().is_null() &&
                !cur.peek_prev().is_null()
            } {
                cur.move_prev();
            }

            cur.move_down();

            while let PeekResult::Data(d) = cur.peek_next() {
                if d < data {
                    cur.move_next();
                } else if d == data {
                    let n = unsafe { cur.ptr_next().unwrap().as_mut() };
                    n.data = data;
                    return;
                } else {
                    break;
                }
            }

            if cur.at_height(h as usize) {
                // Found correct place for ptr[idx]
                let idx = height - h as usize;
                prev[idx] = cur.cur;
                next[idx] = cur.ptr_next();

                h -= 1;
            }
        }

        let node = NonNull::new(self.alloc.alloc_with(|| {
            Node::<T> {
                data,
                next,
                prev,
                height: height as u8
            }
        }));
        
        for i in 0..=1 {
            if let Some(mut p) = prev[i] {
                let p = unsafe { p.as_mut() };
                let idx = ((p.height as i16) - (height as i16) + (i as i16)) as usize;
                p.next[idx] = node;
            } else {
                self.sentinels[height - i] = node;
            }

            if let Some(mut n) = next[i] {
                let n = unsafe { n.as_mut() };
                let idx = ((n.height as i16) - (height as i16) + (i as i16)) as usize;

                n.prev[idx] = node;
            }
        }

    }
}

enum PeekResult<T> {
    Sentinel,
    Data(T),
    Null
}

impl<T> PeekResult<T> {
    pub fn is_null(&self) -> bool {
        if let Self::Null = self {
            true
        } else {
            false
        }
    }
}

impl<'a, T: Copy, const H: usize> Cursor<'a, T, H> {
    pub fn new(skiplift: &'a mut SkipLift<T, H>) -> Self {
        Self {
            lift: skiplift,
            height: None,
            cur: None
        }
    }

    // SAFETY: move_* functions will panic instead of erroring;
    // case on the result of peek_* before calling them
    //
    // This decision was made to improve code legibility based on a 
    // common pattern in algorithms of the skiplift paper, 
    // where we traverse the lift while under a condition
    //
    // TODO: test/reduce overhead of double access, possibly with a
    // conditional peek -> move function

    fn move_next(&mut self) {
        assert!(!self.peek_next().is_null());

        if let Some(cur) = self.cur {
            // We are not at a sentinel
            let cur = unsafe { cur.as_ref() };
            let idx: usize = cur.height as usize - self.height.unwrap();
            self.cur = Some(cur.next[idx].unwrap());
        } else {
            self.cur = Some(self.lift.sentinels[self.height.unwrap()].unwrap());
        }
    }

    fn move_prev(&mut self) {
        assert!(!self.peek_prev().is_null());

        let cur = unsafe { self.cur.unwrap().as_ref() };
        let idx: usize = cur.height as usize - self.height.unwrap();
        self.cur = cur.prev[idx];
    }

    #[allow(dead_code)]
    fn move_up(&mut self) {
        assert!(!self.peek_up().is_null());

        if let Some(cur) = self.cur {
            // We are not at a sentinel
            let cur = unsafe { cur.as_ref() };
            if cur.height as usize != self.height.unwrap() { 
                self.height = Some(self.height.unwrap() + 1);
            } else {
                panic!();
            }
        } else {
            // We are at a sentinel
            self.height = if self.height.unwrap() < H - 1 {
                Some(self.height.unwrap() + 1)
            } else {
                None
            };
        }
    }

    fn move_down(&mut self) {
        assert!(!self.peek_down().is_null());
        
        if let Some(cur) = self.cur {
            let cur = unsafe { cur.as_ref() };
            if cur.height as usize == self.height.unwrap() {
                self.height = Some(self.height.unwrap() - 1);
            } else {
                panic!();
            }
        } else {
            self.height = if let Some(height) = self.height {
                if height > 0 {
                    Some(height - 1)
                } else {
                    panic!()
                }
            } else { Some(H - 1) };
        }
    }

    fn peek_next(&self) -> PeekResult<T> {
        if let Some(cur) = self.cur {
            // We are not at a sentinel
            let cur = unsafe { cur.as_ref() };
            let idx: usize = cur.height as usize - self.height.unwrap();
            if let Some(next) = cur.next[idx] {
                PeekResult::Data(unsafe { 
                    next.as_ref().data 
                })
            } else {
                PeekResult::Null
            }
        } else {
            if let Some(height) = self.height {
                if let Some(next) = self.lift.sentinels[height] {
                    PeekResult::Data(unsafe {
                        next.as_ref().data
                    })
                } else {
                    PeekResult::Null
                }
            } else {
                PeekResult::Null
            }
        }
    }

    fn peek_prev(&self) -> PeekResult<T> {
        if let Some(cur) = self.cur {
            let cur = unsafe { cur.as_ref() };      
            let idx: usize = cur.height as usize - self.height.unwrap();
            if let Some(prev) = cur.prev[idx] {
                PeekResult::Data(unsafe {
                    prev.as_ref().data
                })
            } else {
                PeekResult::Sentinel
            }
        } else {
            PeekResult::Null
        }
    }

    #[allow(dead_code)]
    fn peek_up(&self) -> PeekResult<T> {
        if let Some(cur) = self.cur {
            let cur = unsafe { cur.as_ref() };
            if self.height.unwrap() != cur.height as usize {
                PeekResult::Data(cur.data)
            } else {
                PeekResult::Null
            }
        } else {
            if let Some(_) = self.height {
                PeekResult::Sentinel 
            } else {
                PeekResult::Null
            }
        }
    }

    fn peek_down(&self) -> PeekResult<T> {
        if let Some(cur) = self.cur {
            let cur = unsafe { cur.as_ref() };
            if self.height.unwrap() == cur.height as usize {
                PeekResult::Data(cur.data)
            } else {
                PeekResult::Null
            }
        } else {
            if let Some(height) = self.height {
                if height > 0 { PeekResult::Sentinel } 
                else { PeekResult::Null }
            } else { PeekResult::Sentinel }
        }
    }

    fn ptr_next(&self) -> Link<T> {
        if let Some(cur) = self.cur {
            let cur = unsafe { cur.as_ref() };
            let idx = cur.height as usize - self.height.unwrap();
            cur.next[idx]
        } else {
            if let Some(height) = self.height {
                self.lift.sentinels[height]
            } else { None }
        }
    }

    #[allow(dead_code)]
    fn ptr_prev(&self) -> Link<T> {
        if let Some(cur) = self.cur {
            let cur = unsafe { cur.as_ref() };
            let idx = cur.height as usize - self.height.unwrap();
            cur.prev[idx]
        } else { None }
    }

    fn above_height(&self, height: usize) -> bool {
        if let Some(current_height) = self.height {
            current_height > height
        } else {
            true
        }
    }

    fn at_height(&self, height: usize) -> bool {
        if let Some(current_height) = self.height {
            current_height == height
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SkipLift;

    #[test]
    fn basic_small() {
        let mut lift = SkipLift::<i64, 16>::new();
            
        for _ in 0..1_000_000 {
            lift.upsert_if(1, |x| match x {
                Some(_) => true,
                None => true
            });
        }
    }
}
