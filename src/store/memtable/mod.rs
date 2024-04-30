mod skiplift;

pub use skiplift::SkipLift;

use std::error::Error;
use std::iter::Iterator;
use std::cmp::Ord;

type Res<T> = Result<T, Box<dyn Error>>;

pub trait MemTable<K: Ord, V>: Iterator{
    fn new(size: Option<usize>) -> Self;
    fn create(&self, key: K, val: V) -> Res<()>;
    fn read(&self, key: K) -> Res<V>;
    fn update(&self, key: K, new_val: V) -> Res<V>;
    fn delete(&self, key: K) -> Res<V>;
}

