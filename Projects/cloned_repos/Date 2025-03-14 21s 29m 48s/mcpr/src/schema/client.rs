//! Client-specific MCP schema types

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::common::{
    BlobResourceContents, Cursor, Implementation, LoggingLevel, ProgressToken, Prompt,
    PromptMessage, Resource, ResourceTemplate, Root, TextResourceContents, Tool,
};
use super::json_rpc::RequestId;

/// Client capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Experimental, non-standard capabilities that the client supports.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experimental: Option<HashMap<String, Value>>,

    /// Present if the client supports listing roots.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roots: Option<RootsCapability>,

    /// Present if the client supports sampling from an LLM.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<Value>,
}

/// Roots capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootsCapability {
    /// Whether the client supports notifications for changes to the roots list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// This request is sent from the client to the server when it first connects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeRequest {
    pub method: String,
    pub params: InitializeParams,
}

/// Parameters for initialize request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeParams {
    /// The latest version of the Model Context Protocol that the client supports.
    pub protocol_version: String,

    /// Client capabilities
    pub capabilities: ClientCapabilities,

    /// Client information
    pub client_info: Implementation,
}

/// This notification is sent from the client to the server after initialization has finished.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializedNotification {
    pub method: String,
}

/// A notification which can be sent by either side to indicate that it is cancelling a previously-issued request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelledNotification {
    pub method: String,
    pub params: CancelledParams,
}

/// Parameters for cancelled notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelledParams {
    /// The ID of the request to cancel.
    pub request_id: RequestId,

    /// An optional string describing the reason for the cancellation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// An out-of-band notification used to inform the receiver of a progress update for a long-running request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressNotification {
    pub method: String,
    pub params: ProgressParams,
}

/// Parameters for progress notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressParams {
    /// The progress token which was given in the initial request.
    pub progress_token: ProgressToken,

    /// The progress thus far.
    pub progress: f64,

    /// Total number of items to process (or total progress required), if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
}

/// A ping, issued by either the server or the client, to check that the other party is still alive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingRequest {
    pub method: String,
}

/// Sent from the client to request a list of resources the server has.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResourcesRequest {
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<PaginatedParams>,
}

/// Parameters for paginated requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedParams {
    /// An opaque token representing the current pagination position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<Cursor>,
}

/// The server's response to a resources/list request from the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResourcesResult {
    /// An opaque token representing the pagination position after the last returned result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<Cursor>,

    /// The list of resources
    pub resources: Vec<Resource>,
}

/// Sent from the client to request a list of resource templates the server has.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResourceTemplatesRequest {
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<PaginatedParams>,
}

/// The server's response to a resources/templates/list request from the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResourceTemplatesResult {
    /// An opaque token representing the pagination position after the last returned result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<Cursor>,

    /// The list of resource templates
    pub resource_templates: Vec<ResourceTemplate>,
}

/// Sent from the client to the server, to read a specific resource URI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceRequest {
    pub method: String,
    pub params: ReadResourceParams,
}

/// Parameters for read resource request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceParams {
    /// The URI of the resource to read.
    pub uri: String,
}

/// The server's response to a resources/read request from the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceResult {
    pub contents: Vec<ResourceContent>,
}

/// Resource content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResourceContent {
    Text(TextResourceContents),
    Blob(BlobResourceContents),
}

/// Sent from the client to request resources/updated notifications from the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeRequest {
    pub method: String,
    pub params: SubscribeParams,
}

/// Parameters for subscribe request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeParams {
    /// The URI of the resource to subscribe to.
    pub uri: String,
}

/// Sent from the client to request cancellation of resources/updated notifications from the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsubscribeRequest {
    pub method: String,
    pub params: UnsubscribeParams,
}

/// Parameters for unsubscribe request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsubscribeParams {
    /// The URI of the resource to unsubscribe from.
    pub uri: String,
}

/// Sent from the client to request a list of prompts and prompt templates the server has.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListPromptsRequest {
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<PaginatedParams>,
}

/// The server's response to a prompts/list request from the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListPromptsResult {
    /// An opaque token representing the pagination position after the last returned result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<Cursor>,

    /// The list of prompts
    pub prompts: Vec<Prompt>,
}

/// Used by the client to get a prompt provided by the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptRequest {
    pub method: String,
    pub params: GetPromptParams,
}

/// Parameters for get prompt request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptParams {
    /// The name of the prompt or prompt template.
    pub name: String,

    /// Arguments to use for templating the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<HashMap<String, String>>,
}

/// The server's response to a prompts/get request from the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptResult {
    /// An optional description for the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The prompt messages
    pub messages: Vec<PromptMessage>,
}

/// Sent from the client to request a list of tools the server has.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListToolsRequest {
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<PaginatedParams>,
}

/// The server's response to a tools/list request from the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListToolsResult {
    /// An opaque token representing the pagination position after the last returned result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<Cursor>,

    /// The list of tools
    pub tools: Vec<Tool>,
}

/// Used by the client to invoke a tool provided by the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolRequest {
    pub method: String,
    pub params: CallToolParams,
}

/// Parameters for call tool request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolParams {
    /// The name of the tool to call
    pub name: String,

    /// Arguments for the tool
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<HashMap<String, Value>>,
}

/// A request from the client to the server, to enable or adjust logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetLevelRequest {
    pub method: String,
    pub params: SetLevelParams,
}

/// Parameters for set level request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetLevelParams {
    /// The level of logging that the client wants to receive from the server.
    pub level: LoggingLevel,
}

/// A request from the client to the server, to ask for completion options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteRequest {
    pub method: String,
    pub params: CompleteParams,
}

/// Parameters for complete request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteParams {
    /// Reference to a prompt or resource
    pub ref_: Reference,

    /// The argument's information
    pub argument: ArgumentInfo,
}

/// Reference to a prompt or resource
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Reference {
    Prompt(PromptReference),
    Resource(ResourceReference),
}

/// Identifies a prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptReference {
    pub r#type: String,

    /// The name of the prompt or prompt template
    pub name: String,
}

/// A reference to a resource or resource template definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReference {
    pub r#type: String,

    /// The URI or URI template of the resource.
    pub uri: String,
}

/// Argument information for completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentInfo {
    /// The name of the argument
    pub name: String,

    /// The value of the argument to use for completion matching.
    pub value: String,
}

/// The client's response to a roots/list request from the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListRootsResult {
    pub roots: Vec<Root>,
}

/// A notification from the client to the server, informing it that the list of roots has changed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootsListChangedNotification {
    pub method: String,
}
