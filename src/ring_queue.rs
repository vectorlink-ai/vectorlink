use crate::{bitmap::Bitmap, layer::OrderedFloat};
use std::fmt::Debug;

#[derive(Debug)]
pub enum VecOrSlice<'a, T> {
    Vec(Vec<T>),
    Slice(&'a mut [T]),
}

impl<T> std::ops::Deref for VecOrSlice<'_, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        match self {
            VecOrSlice::Vec(it) => it,
            VecOrSlice::Slice(it) => it,
        }
    }
}

impl<T> std::ops::DerefMut for VecOrSlice<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            VecOrSlice::Vec(it) => it,
            VecOrSlice::Slice(it) => it,
        }
    }
}

pub struct RingQueue<'a> {
    ids: VecOrSlice<'a, u32>,
    priorities: VecOrSlice<'a, f32>,
    head: u32,
    len: u32,
}

impl<'a> RingQueue<'a> {
    fn assert_capacity_len(capacity: usize, len: usize) {
        assert_ne!(capacity, 0, "cannot create a ring queue with capacity 0");
        assert!(
            capacity < u32::MAX as usize,
            "capacity can't exceed u32::MAX-1"
        );
        assert!(len <= capacity);
    }
    pub fn new_with(capacity: usize, ids_slice: &[u32], priorities_slice: &[f32]) -> Self {
        assert_eq!(ids_slice.len(), priorities_slice.len());
        Self::assert_capacity_len(capacity, ids_slice.len());
        let len = ids_slice.len() as u32;
        let mut ids = Vec::with_capacity(capacity);
        ids.extend_from_slice(ids_slice);
        let mut priorities = Vec::with_capacity(capacity);
        priorities.extend_from_slice(priorities_slice);
        unsafe {
            ids.set_len(capacity);
            priorities.set_len(capacity);
        }
        Self {
            ids: VecOrSlice::Vec(ids),
            priorities: VecOrSlice::Vec(priorities),
            head: 0,
            len,
        }
    }
    pub fn new_with_mut_slices(
        len: usize,
        ids_slice: &'a mut [u32],
        priorities_slice: &'a mut [f32],
    ) -> Self {
        // len is starting length
        assert_eq!(ids_slice.len(), priorities_slice.len());
        assert!(len < ids_slice.len());
        Self {
            ids: VecOrSlice::Slice(ids_slice),
            priorities: VecOrSlice::Slice(priorities_slice),
            head: 0,
            len: len as u32,
        }
    }

    #[allow(clippy::uninit_vec)]
    pub fn new(capacity: usize) -> Self {
        Self::assert_capacity_len(capacity, 0);
        unsafe {
            let mut ids = Vec::with_capacity(capacity);
            ids.set_len(capacity);
            ids.shrink_to_fit();

            let mut priorities = Vec::with_capacity(capacity);
            priorities.set_len(capacity);
            priorities.shrink_to_fit();
            Self {
                ids: VecOrSlice::Vec(ids),
                priorities: VecOrSlice::Vec(priorities),
                head: 0,
                len: 0,
            }
        }
    }

    pub fn reinit_from(&mut self, queue: &RingQueue) {
        assert!(self.capacity() >= queue.len());
        self.head = 0;
        self.len = queue.len;
        for (ix, (id, priority)) in queue.iter().enumerate() {
            self.ids[ix] = id;
            self.priorities[ix] = priority;
        }
    }

    #[allow(clippy::uninit_vec)]
    pub fn clone_with_capacity(&self, capacity: usize) -> Self {
        assert!(self.len() <= capacity);
        unsafe {
            let mut ids = Vec::with_capacity(capacity);
            ids.set_len(capacity);
            ids.shrink_to_fit();

            let mut priorities = Vec::with_capacity(capacity);
            priorities.set_len(capacity);
            priorities.shrink_to_fit();
            for (ix, (id, priority)) in self.iter().enumerate() {
                ids[ix] = id;
                priorities[ix] = priority;
            }

            Self {
                ids: VecOrSlice::Vec(ids),
                priorities: VecOrSlice::Vec(priorities),
                head: 0,
                len: self.len,
            }
        }
    }

    #[cfg(test)]
    pub fn print_all(&self) {
        eprintln!("queue presentation: {:?}", self);
        eprintln!("actual ids: {:?}", self.ids);
        eprintln!("actual priorities: {:?}", self.priorities);
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn capacity(&self) -> usize {
        self.ids.len()
    }

    pub fn head(&self) -> usize {
        self.head as usize
    }

    pub fn tail(&self) -> usize {
        (self.head as usize + self.len()) % self.capacity()
    }

    pub fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push_last(&mut self, entry: (u32, f32)) {
        if self.is_full() {
            panic!("tried to push to a queue that is at capacity");
        }

        // if we're not at capacity, we should be able to set tail
        let idx = self.tail();
        self.ids[idx] = entry.0;
        self.priorities[idx] = entry.1;
        self.len += 1;
    }

    fn internal_index(&self, index: usize) -> usize {
        (self.head() + index) % self.capacity()
    }

    pub fn get(&self, index: usize) -> (u32, f32) {
        assert!(
            index < self.len(),
            "tried to retrieve past end of ring queue"
        );
        let idx = self.internal_index(index);
        (self.ids[idx], self.priorities[idx])
    }

    pub fn set(&mut self, index: usize, entry: (u32, f32)) {
        assert!(
            index < self.len(),
            "tried to retrieve past end of ring queue"
        );
        let idx = self.internal_index(index);
        self.ids[idx] = entry.0;
        self.priorities[idx] = entry.1;
    }

    pub fn first(&self) -> (u32, f32) {
        self.get(0)
    }

    pub fn last(&self) -> (u32, f32) {
        self.get(self.len() - 1)
    }

    pub fn pop_first(&mut self) -> (u32, f32) {
        let result = self.first();
        self.head = (self.head + 1) % self.capacity() as u32;
        self.len -= 1;

        result
    }

    pub fn pop_first_n(&mut self, count: usize) -> (Vec<(u32, f32)>, usize) {
        let mut result = Vec::with_capacity(count);
        let mut popped = 0;
        while popped != count && !self.is_empty() {
            result.push(self.pop_first());
            popped += 1;
        }

        (result, popped)
    }

    pub fn insert_at(&mut self, index: usize, entry: (u32, f32)) {
        assert!(index <= self.len(), "tried to insert past end of queue");
        let mut l = self.len();
        // BUG: shouldn't this avoid the push if we're inserting at the very end?
        if l != 0 {
            if !self.is_full() {
                self.push_last(self.last());
            }

            l -= 1;

            while l > index {
                self.set(l, self.get(l - 1));
                l -= 1
            }

            self.set(index, entry);
        } else {
            self.push_last(entry)
        }
    }

    pub fn iter_from(&self, index: usize) -> RingQueueIterator {
        assert!(index <= self.len(), "tried to iterate past end");
        RingQueueIterator { queue: self, index }
    }

    pub fn iter<'b>(&'b self) -> RingQueueIterator<'b> {
        self.iter_from(0)
    }

    pub fn get_all(&self) -> Vec<(u32, f32)> {
        let mut result = Vec::with_capacity(self.len());
        result.extend(self.iter());
        result
    }
}

pub struct RingQueueIterator<'a> {
    queue: &'a RingQueue<'a>,
    index: usize,
}

impl<'a> Iterator for RingQueueIterator<'a> {
    type Item = (u32, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.queue.len() {
            None
        } else {
            let result = self.queue.get(self.index);
            self.index += 1;

            Some(result)
        }
    }
}
impl<'a> Debug for RingQueue<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<head={},len={},capacity={} {:?}>",
            self.head,
            self.len,
            self.capacity(),
            self.get_all()
        )
    }
}

pub struct OrderedRingQueue<'a>(RingQueue<'a>);

impl<'a> OrderedRingQueue<'a> {
    fn assert_ordered<I: Iterator<Item = (u32, f32)>>(mut iter: I) {
        if cfg!(debug_assertions) {
            let pair = iter.next();
            if pair.is_none() {
                return;
            }
            let pair = pair.unwrap();

            let mut last_id = pair.0;
            let mut last_priority = pair.1;
            for (id, priority) in iter {
                assert_ne!(last_id, id);
                assert!(last_priority <= priority);
                if last_priority == priority {
                    assert!(last_id < id);
                }
                last_id = id;
                last_priority = priority;
            }
        }
    }
    pub fn new_with(capacity: usize, ids: &[u32], priorities: &[f32]) -> Self {
        assert_eq!(ids.len(), priorities.len());
        Self::assert_ordered(ids.iter().copied().zip(priorities.iter().copied()));
        Self(RingQueue::new_with(capacity, ids, priorities))
    }
    pub fn new_with_mut_slices(ids_slice: &'a mut [u32], priorities_slice: &'a mut [f32]) -> Self {
        let mut temporary_pairs: Vec<(u32, f32)> = ids_slice
            .iter()
            .zip(&*priorities_slice)
            .map(|(x, y)| (*x, *y))
            .collect();
        temporary_pairs.sort_by_key(|(i, f)| (OrderedFloat(*f), *i));
        let mut ring_queue = Self(RingQueue::new_with_mut_slices(
            0,
            ids_slice,
            priorities_slice,
        ));
        temporary_pairs.iter().for_each(|pair| {
            ring_queue.insert(*pair);
        });
        ring_queue
    }

    pub fn new(capacity: usize) -> Self {
        Self(RingQueue::new(capacity))
    }

    pub fn wrap(queue: RingQueue<'a>) -> Self {
        Self::assert_ordered(queue.iter());
        Self(queue)
    }

    pub fn reinit_from(&mut self, queue: &OrderedRingQueue) {
        self.0.reinit_from(&queue.0)
    }

    pub fn clone_with_capacity(&self, capacity: usize) -> Self {
        Self(self.0.clone_with_capacity(capacity))
    }

    #[cfg(test)]
    pub fn print_all(&self) {
        self.0.print_all();
    }

    pub fn head(&self) -> usize {
        self.0.head()
    }

    pub fn tail(&self) -> usize {
        self.0.tail()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    pub fn is_full(&self) -> bool {
        self.0.is_full()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, index: usize) -> (u32, f32) {
        self.0.get(index)
    }

    pub fn first(&self) -> (u32, f32) {
        self.0.first()
    }

    pub fn last(&self) -> (u32, f32) {
        self.0.last()
    }

    pub fn pop_first(&mut self) -> (u32, f32) {
        self.0.pop_first()
    }

    pub fn pop_first_n(&mut self, count: usize) -> (Vec<(u32, f32)>, usize) {
        self.0.pop_first_n(count)
    }

    pub fn set(&mut self, index: usize, entry: (u32, f32)) {
        self.0.set(index, entry)
    }

    pub fn iter_from(&self, index: usize) -> RingQueueIterator {
        self.0.iter_from(index)
    }

    pub fn iter(&self) -> RingQueueIterator {
        self.0.iter()
    }

    pub fn get_all(&self) -> Vec<(u32, f32)> {
        self.0.get_all()
    }

    pub fn insert(&mut self, elt: (u32, f32)) -> bool {
        let mut did_something = false;
        let i = self.insertion_point_from(elt, 0);
        if i == self.capacity() {
            return false;
        }
        let val_at_i = self.get(i);
        // only insert if we aren't identical
        if val_at_i.0 != elt.0 || val_at_i.1 != elt.1 {
            did_something = true;
            self.0.insert_at(i, elt);
        }
        did_something
    }

    #[cfg(test)]
    fn insertion_point(&self, pair: (u32, f32)) -> usize {
        self.insertion_point_from(pair, 0)
    }

    fn insertion_point_from(&self, pair: (u32, f32), start_from: usize) -> usize {
        let l = self.len();
        if l == 0 {
            return 0;
        }

        let mut low = start_from;
        let mut high = l;
        let mut size = high - low;
        while low < high {
            let mut mid = low + size / 2;
            let mut point = self.get(mid);

            if point.1 < pair.1 {
                low = mid + 1;
            } else if point.1 > pair.1 {
                high = mid;
            } else {
                // we must retreat as long as the id is lower and priority the same
                while mid > low && pair.0 < point.0 && pair.1 == point.1 {
                    let peek = self.get(mid - 1);
                    if peek.1 != pair.1 {
                        break;
                    }
                    point = peek;
                    mid -= 1;
                }
                // we must advance as long as the id is higher and priority the same
                while pair.0 > point.0 && point.1 == pair.1 {
                    mid += 1;
                    if l == mid {
                        return l;
                    }
                    point = self.get(mid);
                }

                return mid;
            }

            size = high - low;
        }

        low
    }

    pub fn merge(&mut self, other: &OrderedRingQueue) -> bool {
        let mut did_something = false;
        let mut last_idx = 0;

        for elt in other.iter() {
            if last_idx > self.capacity() {
                break;
            }
            let i = self.insertion_point_from(elt, last_idx);
            if i == self.capacity() {
                break;
            }
            let val_at_i = self.get(i);
            // only insert if we aren't identical
            if val_at_i.0 != elt.0 || val_at_i.1 != elt.1 {
                did_something = true;
                self.0.insert_at(i, elt);
            }
            last_idx = i;
        }

        did_something
    }
}

impl<'a> Debug for OrderedRingQueue<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

pub fn ring_double_insert(
    visit_queue: &mut OrderedRingQueue,
    search_queue: &mut OrderedRingQueue,
    ids: &[u32],
    priorities: &[f32],
    seen: &Bitmap,
) -> bool {
    assert!(ids.len() == priorities.len());
    let mut did_something = false;
    if ids.is_empty() {
        return false;
    }
    for i2 in 0..ids.len() {
        let id = ids[i2];
        // skip if the id is out of band
        if id == u32::MAX || seen.check(id as usize) {
            continue;
        }

        let first = (id, priorities[i2]);
        let i = visit_queue.insertion_point_from(first, 0);
        if i != visit_queue.capacity() {
            if i != visit_queue.len() {
                let val_at_i = visit_queue.get(i);
                // only insert if we aren't identical
                if val_at_i.0 != first.0 {
                    visit_queue.0.insert_at(i, first);
                }
            } else {
                visit_queue.0.insert_at(i, first);
            }
        }

        let i = search_queue.insertion_point_from(first, 0);
        if i != search_queue.capacity() {
            if i != search_queue.len() {
                let val_at_i = search_queue.get(i);
                // only insert if we aren't identical
                if val_at_i.0 != first.0 {
                    did_something = true;
                    search_queue.0.insert_at(i, first);
                }
            } else {
                did_something = true;
                search_queue.0.insert_at(i, first);
            }
        }
    }

    did_something
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn load_and_retrieve_empty_buffer() {
        let queue = RingQueue::new(100);
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn load_and_retrieve_non_empty_buffer() {
        let queue = RingQueue::new_with(100, &[42], &[0.3]);
        assert!(!queue.is_empty());
        let first = queue.first();
        assert_eq!(first, (42, 0.3));
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn load_and_pop() {
        let mut queue = RingQueue::new_with(100, &[42], &[0.3]);
        let first = queue.pop_first();
        assert_eq!(first, (42, 0.3));
        assert!(queue.is_empty());
    }

    #[test]
    fn load_and_push() {
        let mut queue = RingQueue::new(10);

        for i in 0..10 {
            queue.push_last((i, i as f32 / 10.0));
        }
        assert_eq!(queue.len(), 10);
        assert!(queue.is_full());
        assert!(!queue.is_empty());

        for i in 0..10 {
            let first = queue.pop_first();
            assert_eq!(first, (i, i as f32 / 10.0));
            eprintln!("{:?}", queue);
            assert_eq!(queue.len(), 9 - i as usize);
        }
    }

    #[test]
    fn load_and_wrapping_push() {
        let mut queue = RingQueue::new(10);
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
        assert!(!queue.is_full());

        for i in 0..5 {
            queue.push_last((i, i as f32 / 10.0));
        }
        assert_eq!(queue.len(), 5);

        for i in 0..5 {
            assert_eq!(queue.get(i as usize), (i, i as f32 / 10.0));
        }

        for i in 0..5 {
            let first = queue.pop_first();
            assert_eq!(first, (i, i as f32 / 10.0));
            eprintln!("{:?}", queue);
            assert_eq!(queue.len(), 4 - i as usize);
        }
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
        assert!(!queue.is_full());

        for i in 5..10 {
            queue.push_last((i, i as f32 / 10.0));
        }
        assert_eq!(queue.len(), 5);

        for i in 5..10 {
            assert_eq!(queue.get(i as usize - 5), (i, i as f32 / 10.0));
        }
        assert_eq!(queue.get_all(), queue.clone_with_capacity(10).get_all());
        assert_eq!(queue.get_all(), queue.clone_with_capacity(11).get_all());

        for i in 10..15 {
            queue.push_last((i, i as f32 / 10.0));
        }

        assert_eq!(
            queue.get_all(),
            (5..15).map(|i| (i, i as f32 / 10.0)).collect::<Vec<_>>()
        );

        assert_eq!(queue.get_all(), queue.clone_with_capacity(10).get_all());
        assert_eq!(queue.get_all(), queue.clone_with_capacity(11).get_all());

        for i in 5..15 {
            let first = queue.pop_first();
            assert_eq!(first, (i, i as f32 / 10.0));
            eprintln!("{:?}", queue);
            assert_eq!(queue.len(), 14 - i as usize);
        }

        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
        assert!(!queue.is_full());
        assert_eq!(queue.get_all(), queue.clone_with_capacity(10).get_all());
        assert_eq!(queue.get_all(), queue.clone_with_capacity(11).get_all());
    }

    #[test]
    fn insertion_points() {
        let queue = OrderedRingQueue::new_with(10, &[2, 4, 6, 8, 10], &[1.0, 2.0, 3.0, 4.0, 5.0]);

        // below first
        let idx = queue.insertion_point((5, 0.0));
        assert_eq!(idx, 0);

        // advanced by id
        let idx = queue.insertion_point((5, 2.0));
        assert_eq!(idx, 2);

        // exact match
        let idx = queue.insertion_point((4, 2.0));
        assert_eq!(idx, 1);

        // lower by id
        let idx = queue.insertion_point((3, 2.0));
        eprintln!("{queue:?}");
        assert_eq!(idx, 1);

        // priority higher
        let idx = queue.insertion_point((12, 3.5));
        assert_eq!(idx, 3);

        // priority past the end
        let idx = queue.insertion_point((12, 6.0));
        assert_eq!(idx, 5);
    }

    #[test]
    fn insertion_point_identical() {
        let queue = OrderedRingQueue::new_with(5, &[2, 4, 6, 8, 10], &[1.0, 2.0, 3.0, 4.0, 5.0]);

        // priority over boundary
        let idx = queue.insertion_point((2, 1.0));

        assert_eq!(idx, 0);
    }

    #[test]
    fn insertion_wrapped_points() {
        let mut queue = RingQueue::new_with(5, &[2, 4, 6, 8, 10], &[1.0, 2.0, 3.0, 4.0, 5.0]);

        assert!(queue.is_full());

        // force ring to wrap.
        for _i in 0..3 {
            queue.pop_first();
        }

        for i in 6..9 {
            queue.push_last((i * 2, i as f32 * 1.0));
        }

        let queue = OrderedRingQueue::wrap(queue);

        eprintln!("before insertion: {:?}", queue.get_all());

        // priority over boundary
        let idx = queue.insertion_point((12, 5.5));
        assert_eq!(idx, 2);

        // priority over boundary
        let idx = queue.insertion_point((13, 6.0));
        assert_eq!(idx, 3);

        // priority over boundary
        let idx = queue.insertion_point((11, 6.0));
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_insert_at() {
        let mut queue = RingQueue::new_with(5, &[2, 4, 6, 8, 10], &[1.0, 2.0, 3.0, 4.0, 5.0]);

        let p = (3, 1.5);
        queue.insert_at(1, p);

        assert_eq!(queue.get(1), p);
        assert_eq!(queue.get(2), (4, 2.0));
    }

    #[test]
    fn merge_no_wrap() {
        let mut queue =
            OrderedRingQueue::new_with(5, &[2, 4, 6, 8, 10], &[1.0, 2.0, 3.0, 4.0, 5.0]);

        let i = queue.insertion_point((4, 2.0));
        eprintln!("and we are to insert the duplicate at: {i}");

        let queue1 = OrderedRingQueue::new_with(5, &[4, 8, 9, 11], &[2.0, 4.0, 4.5, 5.5]);

        queue.merge(&queue1);

        assert_eq!(
            vec![(2, 1.0), (4, 2.0), (6, 3.0), (8, 4.0), (9, 4.5),],
            queue.get_all()
        );
    }

    #[test]
    fn test_pop_first_n() {
        let mut queue = RingQueue::new_with(5, &[2, 4, 6, 8, 10], &[1.0, 2.0, 3.0, 4.0, 5.0]);

        let (results, actual_pops) = queue.pop_first_n(3);
        assert_eq!(actual_pops, 3);

        assert_eq!(queue.get_all(), vec![(8, 4.0), (10, 5.0)]);

        assert_eq!(results, vec![(2, 1.0), (4, 2.0), (6, 3.0),]);
    }

    #[test]
    fn double_insert() {
        let mut visit_queue = OrderedRingQueue::new(5);
        let mut candidates_queue =
            OrderedRingQueue::new_with(5, &[2, 4, 6, 8, 10], &[2.0, 4.0, 6.0, 8.0, 10.0]);
        let ids = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let priorities: Vec<_> = ids.iter().map(|x| *x as f32).collect();
        let seen = Bitmap::new(20);
        ring_double_insert(
            &mut visit_queue,
            &mut candidates_queue,
            &ids,
            &priorities,
            &seen,
        );

        assert_eq!(
            visit_queue.get_all(),
            vec![(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0),]
        );

        assert_eq!(
            candidates_queue.get_all(),
            vec![(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0),]
        );
    }

    #[test]
    fn double_insert_zero() {
        let mut visit_queue = OrderedRingQueue::new_with(5, &[0], &[0.0]);
        eprintln!("visit queue:");
        visit_queue.print_all();
        let mut candidates_queue = OrderedRingQueue::new_with(5, &[0], &[0.0]);
        eprintln!("candidates queue:");
        candidates_queue.print_all();
        let ids = vec![3, 0];
        let seen = Bitmap::new(20);
        let priorities = vec![0.0, 0.0];
        ring_double_insert(
            &mut visit_queue,
            &mut candidates_queue,
            &ids,
            &priorities,
            &seen,
        );

        assert_eq!(visit_queue.get_all(), vec![(0, 0.0), (3, 0.0)]);
    }

    #[test]
    fn double_insert_garbage() {
        let mut visit_queue = OrderedRingQueue::new_with(5, &[0], &[0.0]);
        let mut candidates_queue = OrderedRingQueue::new_with(5, &[0], &[0.0]);
        let ids = vec![3, 0, u32::MAX];
        let seen = Bitmap::new(20);
        let priorities = vec![0.0, 0.0, f32::INFINITY];
        ring_double_insert(
            &mut visit_queue,
            &mut candidates_queue,
            &ids,
            &priorities,
            &seen,
        );

        assert_eq!(visit_queue.get_all(), vec![(0, 0.0), (3, 0.0)]);
    }
}
