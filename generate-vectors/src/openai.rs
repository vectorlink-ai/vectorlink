use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::Duration,
};

use anyhow::Context;
use bytes::Bytes;
use clap::ValueEnum;
use lazy_static::lazy_static;
use regex::Regex;
use reqwest::{header::HeaderValue, Body, Client, Method, Request, StatusCode, Url};
use serde::{
    de::{SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use thiserror::Error;
use tiktoken_rs::{cl100k_base, CoreBPE};
use tokio::sync::{Mutex, Notify, RwLock};

use crate::model::Model;

pub struct OpenAIEmbedder {
    model: OpenAIModelType,
    api_key: String,
}

impl OpenAIEmbedder {
    pub fn new(model: OpenAIModelType, api_key: String) -> Self {
        Self { model, api_key }
    }
}

const OPENAI_EMBEDDING_LENGTH: usize = 1536 * 4;
impl Model for OpenAIEmbedder {
    async fn embeddings_for(
        &self,
        strings: &[String],
        output: &mut [u8],
    ) -> Result<(), anyhow::Error> {
        embeddings_for_output(&self.api_key, strings, self.model, output)
            .await
            .context("could not generate ada2 embeddings from openai")
    }
}

type OpenAIStandardEmbedding = [f32; 1536];

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [Vec<usize>],
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
struct EmbeddingResponse {
    object: String,
    data: Vec<EmbeddingData>,
    model: String,
    usage: EmbeddingUsage,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
struct EmbeddingData {
    object: String,
    index: usize,
    #[serde(deserialize_with = "deserialize_single_embedding")]
    embedding: OpenAIStandardEmbedding,
}

fn deserialize_single_embedding<'de, D>(
    deserializer: D,
) -> Result<OpenAIStandardEmbedding, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(SingleEmbeddingVisitor)
}

struct SingleEmbeddingVisitor;

impl<'de> Visitor<'de> for SingleEmbeddingVisitor {
    type Value = OpenAIStandardEmbedding;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "a list of 1536 floats")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut result = [0.0; 1536];
        let mut index = 0;
        while let Some(next) = seq.next_element()? {
            if index >= result.len() {
                // should not really happen but let's not panic
                break;
            }
            result[index] = next;
            index += 1;
        }

        Ok(result)
    }
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
struct EmbeddingUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("error while doing openai request: {0:?}")]
    ReqwestError(#[from] reqwest::Error),
    #[error("response had bad status code: {}", .0)]
    BadStatus(StatusCode, String),
    #[error("incomplete body")]
    IncompleteBody,

    #[error("error while parsing json: {0:?}")]
    BadJson(serde_json::Error, String),
}

lazy_static! {
    static ref ENCODER: CoreBPE = cl100k_base().unwrap();
}

fn tokens_for(s: &str) -> Vec<usize> {
    ENCODER.encode_with_special_tokens(s)
}

const MAX_TOKEN_COUNT: usize = 8191;
fn truncated_tokens_for(s: &str) -> Vec<usize> {
    let mut tokens = tokens_for(s);
    if tokens.len() > MAX_TOKEN_COUNT {
        tokens.truncate(MAX_TOKEN_COUNT);
    }

    tokens
}

struct RateLimiter {
    budget: Arc<Mutex<usize>>,
    waiters: Arc<Mutex<VecDeque<(usize, Arc<Notify>)>>>,
}

impl RateLimiter {
    fn new(budget: usize) -> Self {
        Self {
            budget: Arc::new(Mutex::new(budget)),
            waiters: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    async fn wakeup_existing(mut budget: usize, waiters: &mut VecDeque<(usize, Arc<Notify>)>) {
        while waiters
            .front()
            .map(|(requested_budget, _)| *requested_budget < budget)
            .unwrap_or(false)
        {
            eprintln!("wake up time!");
            let (requested_budget, wakeup) = waiters.pop_front().unwrap();
            wakeup.notify_one();
            budget -= requested_budget;
        }
    }

    async fn budget_tokens(&self, requested_budget: usize) {
        loop {
            let mut budget = self.budget.lock().await;
            if requested_budget <= *budget {
                *budget -= requested_budget;
                eprintln!("requested {}. budget now {}", requested_budget, *budget);
                let inner_budget = self.budget.clone();
                let inner_waiters = self.waiters.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    let mut budget = inner_budget.lock().await;
                    *budget += requested_budget;
                    let budget_copy = *budget;
                    std::mem::drop(budget);
                    eprintln!("minute passed. budget now {}", budget_copy);
                    Self::wakeup_existing(budget_copy, &mut *inner_waiters.lock().await).await;
                });
                return;
            } else {
                eprintln!("rate limit time!");
                std::mem::drop(budget);
                let notify = Arc::new(Notify::new());
                {
                    let mut waiters = self.waiters.lock().await;
                    waiters.push_back((requested_budget, notify.clone()));
                }
                notify.notified().await;
            }
        }
    }
}

async fn execute_request_and_get_bytes(
    client: &Client,
    req: Request,
) -> Result<(StatusCode, Bytes), reqwest::Error> {
    let client_response = client.execute(req).await?;
    let status = client_response.status();
    let bytes = client_response.bytes().await?;
    Ok((status, bytes))
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
pub enum OpenAIModelType {
    Ada2,
    Small3,
}

impl OpenAIModelType {
    fn name(self) -> &'static str {
        match self {
            Self::Ada2 => "text-embedding-ada-002",
            Self::Small3 => "text-embedding-3-small",
        }
    }
}

const MAX_FAILURE_COUNT: usize = 5;
pub async fn embeddings_for(
    api_key: &str,
    strings: &[String],
    model: OpenAIModelType,
) -> Result<(Vec<OpenAIStandardEmbedding>, usize), EmbeddingError> {
    const RATE_LIMIT: usize = 1_000_000;
    const CHUNK_SIZE: usize = 2048;

    lazy_static! {
        static ref ENDPOINT: Url = Url::parse("https://api.openai.com/v1/embeddings").unwrap();
        static ref CLIENT: Client = Client::new();
        static ref LIMITERS: Arc<RwLock<HashMap<String, RateLimiter>>> =
            Arc::new(RwLock::new(HashMap::new()));
    }

    let mut global_failure_count = 0;
    let mut results = Vec::with_capacity(strings.len());
    for strings in strings.chunks(CHUNK_SIZE) {
        {
            let read_guard = LIMITERS.read().await;
            if !read_guard.contains_key(api_key) {
                std::mem::drop(read_guard);
                let mut write_guard = LIMITERS.write().await;
                let limiter = RateLimiter::new(RATE_LIMIT);
                write_guard.insert(api_key.to_owned(), limiter);
            }
        }
        let read_guard = LIMITERS.read().await;
        let limiter = &read_guard[api_key];
        //eprintln!("{strings:?}");
        assert_eq!(None, strings.iter().position(|s| s.is_empty()), "die");

        let token_lists: Vec<_> = strings.iter().map(|s| truncated_tokens_for(s)).collect();

        limiter
            .budget_tokens(token_lists.iter().map(|ts| ts.len()).sum())
            .await;

        let body = EmbeddingRequest {
            model: model.name(),
            input: &token_lists,
            user: None,
        };

        let response: EmbeddingResponse;
        let mut failure_count = 0;
        loop {
            let mut req = Request::new(Method::POST, ENDPOINT.clone());
            let headers = req.headers_mut();
            headers.insert("Content-Type", HeaderValue::from_static("application/json"));
            headers.insert(
                "Authorization",
                HeaderValue::from_str(&format!("Bearer {api_key}")).unwrap(),
            );
            let body_vec = serde_json::to_vec(&body).unwrap();
            let body: Body = body_vec.into();
            *req.body_mut() = Some(body); // once told me the world is gonna roll me

            let result = execute_request_and_get_bytes(&CLIENT, req).await;
            if result.is_err() {
                // something
                if failure_count == MAX_FAILURE_COUNT {
                    return Err(EmbeddingError::IncompleteBody);
                } else {
                    failure_count += 1;
                    eprintln!(
                    "encountered failure {failure_count} while calling openai. retrying.. (incomplete response)"
                );
                    let backoff = 2_u64.pow(failure_count as u32);
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    continue;
                }
            }
            let (status, response_bytes) = result.unwrap();
            if status != StatusCode::OK {
                let body = String::from_utf8_lossy(&response_bytes).to_string();
                if failure_count == MAX_FAILURE_COUNT {
                    return Err(EmbeddingError::BadStatus(status, body));
                } else {
                    failure_count += 1;
                    eprintln!(
                    "encountered failure {failure_count} while calling openai. retrying..\n{body}"
                );
                    let backoff = 2_u64.pow(failure_count as u32);
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    continue;
                }
            }
            match serde_json::from_slice(&response_bytes) {
                Ok(r) => response = r,
                Err(e) => {
                    let body = String::from_utf8_lossy(&response_bytes).to_string();
                    return Err(EmbeddingError::BadJson(e, body));
                }
            }
            break;
        }
        for embedding in response.data {
            results.push(embedding.embedding);
        }
        global_failure_count += failure_count;
    }
    Ok((results, global_failure_count))
}

pub async fn embeddings_for_output(
    api_key: &str,
    strings: &[String],
    model: OpenAIModelType,
    output: &mut [u8],
) -> Result<(), EmbeddingError> {
    assert_eq!(output.len(), 1536 * 4 * strings.len());
    let (embeddings, _failure_count) = embeddings_for(api_key, strings, model).await?;

    for (ix, embedding) in embeddings.into_iter().enumerate() {
        let slice_cast = unsafe {
            std::slice::from_raw_parts(embedding.as_ptr() as *const u8, OPENAI_EMBEDDING_LENGTH)
        };
        let offset = ix * OPENAI_EMBEDDING_LENGTH;
        output[offset..offset + OPENAI_EMBEDDING_LENGTH].copy_from_slice(slice_cast);
    }

    Ok(())
}

pub struct OpenAIDecider {
    api_key: String,
    entity_description: String,
}

impl OpenAIDecider {
    pub fn new(api_key: String, entity_description: String) -> Self {
        Self {
            api_key,
            entity_description,
        }
    }

    pub async fn decide(&self, e1: &str, e2: &str) -> Result<bool, anyhow::Error> {
        lazy_static! {
            static ref ENDPOINT: Url =
                Url::parse("https://api.openai.com/v1/chat/completions").unwrap();
            static ref CLIENT: Client = Client::new();
            static ref YES_MATCHER: Regex = Regex::new(r"yes").unwrap();
            static ref NO_MATCHER: Regex = Regex::new(r"no").unwrap();
        }

        let messages = vec![
            Message {
                role: "system".to_string(),
                content: format!("You are a classifier deciding if two entities are a match or not. These entities are about the following:
{}
Tell me whether the following two records are referring to the same entity or a different entity using a chain of reasoning followed by a single yes or no answer on a single line, without any formatting.
", self.entity_description),
            },
            Message {
                role: "user".to_string(),
                content: format!("1: {e1}
2: {e2}"),
            },
        ];

        let request = CompletionRequest {
            messages,
            model: "gpt-4o".to_string(),
        };
        let mut req = Request::new(Method::POST, ENDPOINT.clone());
        let headers = req.headers_mut();
        headers.insert("Content-Type", HeaderValue::from_static("application/json"));
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {}", self.api_key)).unwrap(),
        );
        let body_vec = serde_json::to_vec(&request).unwrap();
        let body: Body = body_vec.into();
        *req.body_mut() = Some(body);

        let (status, bytes) = execute_request_and_get_bytes(&CLIENT, req).await?;

        let response: PartialCompletionResponse =
            serde_json::from_slice(&bytes).context("could not parse openai completion response")?;

        let message = &response.choices[0].message;
        let last_line = message.content.lines().last().unwrap().to_lowercase();
        if YES_MATCHER.is_match(last_line.as_str()) {
            Ok(true)
        } else if NO_MATCHER.is_match(last_line.as_str()) {
            Ok(false)
        } else {
            eprintln!("got a completion that was not a yes or no: {last_line}");
            Ok(false)
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct CompletionRequest {
    model: String,
    messages: Vec<Message>,
}

#[derive(Deserialize)]
struct PartialCompletionResponse {
    choices: Vec<PartialChoice>,
}

#[derive(Deserialize)]
struct PartialChoice {
    message: Message,
}
