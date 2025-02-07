use std::{collections::HashMap, path::Path};

use crate::templates::ID_FIELD_NAME;
use either::Either;
use vectorlink_hnsw::vectors::Vectors;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct FullGraph {
    fields: HashMap<String, Graph>,
}

impl FullGraph {
    pub fn new(fields: Vec<(String, Graph)>) -> Self {
        let fields: HashMap<String, Graph> = fields.into_iter().collect();
        Self { fields }
    }

    pub fn get(&self, field: &str) -> Option<&Graph> {
        self.fields.get(field)
    }

    pub fn fields(&self) -> Vec<&str> {
        self.fields.keys().map(|s| s.as_str()).collect()
    }

    pub fn id_graph(&self) -> &Graph {
        &self.fields[ID_FIELD_NAME]
    }

    pub fn record_id_field_value(&self, id: u32) -> &str {
        self.id_graph()
            .record_id_to_value(id)
            .expect("Missing id field")
    }

    pub fn record_count(&self) -> usize {
        self.id_graph().values.len()
    }

    pub fn load_vecs<P: AsRef<Path>>(&self, vector_path: P) -> HashMap<String, Vectors> {
        self.fields()
            .iter()
            .filter(|name| **name != ID_FIELD_NAME)
            .map(|name| {
                (
                    name.to_string(),
                    Vectors::load(&vector_path, name)
                        .unwrap_or_else(|_| panic!("Unable to load vector file for {name}")),
                )
            })
            .collect()
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Graph {
    pub values: Vec<String>,
    record_to_value: HashMap<u32, u32>,
    value_to_records: HashMap<u32, Vec<u32>>,
}

impl Graph {
    pub fn new<'a, I: Iterator<Item = &'a str>>(iter: I) -> Self {
        let mut pairs: Vec<(u32, &str)> = iter.enumerate().map(|(i, s)| (i as u32, s)).collect();
        // ensure ordering is random
        pairs.sort_by(|elt1, elt2| {
            gxhash::gxhash64(elt1.1.as_bytes(), 0x533D)
                .cmp(&gxhash::gxhash64(elt2.1.as_bytes(), 0x533D))
        });
        let mut values: Vec<String> = Vec::new();
        let record_to_value: HashMap<u32, u32> = pairs
            .chunk_by(|x, y| x.1 == y.1)
            .flat_map(|slice| {
                assert!(!slice.is_empty());
                if slice[0].1.is_empty() {
                    Either::Left(std::iter::empty())
                } else {
                    let vector_id = values.len() as u32;
                    values.push(slice[0].1.to_owned());
                    Either::Right(
                        slice
                            .iter()
                            .map(move |(record_id, _)| (*record_id, vector_id)),
                    )
                }
            })
            .collect();

        let mut value_to_record: HashMap<u32, Vec<u32>> = HashMap::new();
        for (&record_id, &value_id) in record_to_value.iter() {
            match value_to_record.entry(value_id) {
                std::collections::hash_map::Entry::Occupied(mut occupied_entry) => {
                    occupied_entry.get_mut().push(record_id)
                }
                std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(vec![record_id]);
                }
            }
        }

        Self {
            values,
            record_to_value,
            value_to_records: value_to_record,
        }
    }

    pub fn value_id_to_record_ids(&self, value_id: u32) -> &[u32] {
        self.value_to_records
            .get(&value_id)
            .expect("No corresponding record for value!")
    }

    pub fn record_id_to_value_id(&self, record_id: u32) -> Option<u32> {
        self.record_to_value.get(&record_id).copied()
    }

    pub fn record_id_to_value(&self, record_id: u32) -> Option<&str> {
        self.record_id_to_value_id(record_id)
            .map(|id| self.values[id as usize].as_str())
    }
}

pub struct CompareGraph<'a> {
    pub graph: &'a FullGraph,
    pub vecs: HashMap<String, Vectors>,
}

impl<'a> CompareGraph<'a> {
    pub fn new(graph: &'a FullGraph, vecs: HashMap<String, Vectors>) -> Self {
        Self { graph, vecs }
    }

    pub fn record_id_field_value(&self, id: u32) -> &str {
        self.graph
            .id_graph()
            .record_id_to_value(id)
            .expect("Missing id field")
    }

    pub fn get_vectors(&self, field: &str) -> Option<&Vectors> {
        self.vecs.get(field)
    }
}

#[cfg(test)]
mod tests {
    use super::Graph;

    #[test]
    fn construct_graph() {
        let strings = ["cow", "chicken", "duck", "cow", "", "horse"];
        let graph = Graph::new(strings.into_iter());

        assert_eq!(vec!["chicken", "cow", "duck", "horse"], graph.values);

        assert_eq!(graph.record_to_value[&0], 1);
        assert_eq!(graph.record_to_value[&1], 0);
        assert_eq!(graph.record_to_value[&2], 2);
        assert_eq!(graph.record_to_value[&3], 1);
        assert_eq!(graph.record_to_value.get(&4), None);
        assert_eq!(graph.record_to_value[&5], 3);

        assert_eq!(graph.value_to_records[&0], vec![1]);
        let mut records = graph.value_to_records[&1].clone();
        records.sort();
        assert_eq!(records, vec![0, 3]);
        assert_eq!(graph.value_to_records[&2], vec![2]);
        assert_eq!(graph.value_to_records[&3], vec![5]);
    }
}
