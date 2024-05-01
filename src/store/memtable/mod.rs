mod skiplift;

pub use skiplift::SkipLift;

//use std::iter::IntoIterator;
use std::cmp::PartialOrd;

#[derive(Debug)]
pub struct NotFound;

#[derive(Debug)]
pub struct Collision;

// The default value of V should be the tombstone value
pub trait MemTable<K: PartialOrd, V> {
    fn new(size: Option<usize>) -> Self;
    fn create(&mut self, key: K, val: V) -> Result<(), Collision>;
    fn read(&mut self, key: K) -> Result<V, NotFound>;
    fn update(&mut self, key: K, new_val: V) -> Result<V, NotFound>;
}

