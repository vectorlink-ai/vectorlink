pub struct RingQueue {
    ids: Vec<u32>,
    priorities: Vec<f32>,
    head: usize,
    tail: usize,
    full: bool,
}

impl RingQueue {
    pub fn new(capacity: usize) -> Self {
        assert_ne!(capacity, 0, "cannot create a ring queue with capacity 0");
        unsafe {
            let mut ids = Vec::with_capacity(capacity);
            ids.set_len(capacity);
            ids.shrink_to_fit();

            let mut priorities = Vec::with_capacity(capacity);
            priorities.set_len(capacity);
            priorities.shrink_to_fit();
            Self {
                ids,
                priorities,
                head: 0,
                tail: 0,
                full: false,
            }
        }
    }
}
