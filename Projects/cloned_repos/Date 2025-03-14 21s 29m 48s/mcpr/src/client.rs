//! High-level client implementation for MCP

use crate::{
    constants::LATEST_PROTOCOL_VERSION,
    error::MCPError,
    schema::json_rpc::{JSONRPCMessage, JSONRPCRequest, RequestId},
    transport::Transport,
};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;

/// High-level MCP client
pub struct Client<T: Transport> {
    transport: T,
    next_request_id: i64,
}

impl<T: Transport> Client<T> {
    /// Create a new MCP client with the given transport
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            next_request_id: 1,
        }
    }

    /// Initialize the client
    pub fn initialize(&mut self) -> Result<Value, MCPError> {
        // Start the transport
        self.transport.start()?;

        // Send initialization request
        let initialize_request = JSONRPCRequest::new(
            self.next_request_id(),
            "initialize".to_string(),
            Some(serde_json::json!({
                "protocol_version": LATEST_PROTOCOL_VERSION
            })),
        );

        let message = JSONRPCMessage::Request(initialize_request);
        self.transport.send(&message)?;

        // Wait for response
        let response: JSONRPCMessage = self.transport.receive()?;

        match response {
            JSONRPCMessage::Response(resp) => Ok(resp.result),
            JSONRPCMessage::Error(err) => Err(MCPError::Protocol(format!(
                "Initialization failed: {:?}",
                err
            ))),
            _ => Err(MCPError::Protocol("Unexpected response type".to_string())),
        }
    }

    /// Call a tool on the server
    pub fn call_tool<P: Serialize, R: DeserializeOwned>(
        &mut self,
        tool_name: &str,
        params: &P,
    ) -> Result<R, MCPError> {
        // Create tool call request
        let tool_call_request = JSONRPCRequest::new(
            self.next_request_id(),
            "tool_call".to_string(),
            Some(serde_json::json!({
                "name": tool_name,
                "parameters": serde_json::to_value(params)?
            })),
        );

        let message = JSONRPCMessage::Request(tool_call_request);
        self.transport.send(&message)?;

        // Wait for response
        let response: JSONRPCMessage = self.transport.receive()?;

        match response {
            JSONRPCMessage::Response(resp) => {
                // Extract the tool result from the response
                let result_value = resp.result;
                let result = result_value.get("result").ok_or_else(|| {
                    MCPError::Protocol("Missing 'result' field in response".to_string())
                })?;

                // Parse the result
                serde_json::from_value(result.clone()).map_err(|e| MCPError::Serialization(e))
            }
            JSONRPCMessage::Error(err) => {
                Err(MCPError::Protocol(format!("Tool call failed: {:?}", err)))
            }
            _ => Err(MCPError::Protocol("Unexpected response type".to_string())),
        }
    }

    /// Shutdown the client
    pub fn shutdown(&mut self) -> Result<(), MCPError> {
        // Send shutdown request
        let shutdown_request =
            JSONRPCRequest::new(self.next_request_id(), "shutdown".to_string(), None);

        let message = JSONRPCMessage::Request(shutdown_request);
        self.transport.send(&message)?;

        // Wait for response
        let response: JSONRPCMessage = self.transport.receive()?;

        match response {
            JSONRPCMessage::Response(_) => {
                // Close the transport
                self.transport.close()?;
                Ok(())
            }
            JSONRPCMessage::Error(err) => {
                Err(MCPError::Protocol(format!("Shutdown failed: {:?}", err)))
            }
            _ => Err(MCPError::Protocol("Unexpected response type".to_string())),
        }
    }

    /// Generate the next request ID
    fn next_request_id(&mut self) -> RequestId {
        let id = self.next_request_id;
        self.next_request_id += 1;
        RequestId::Number(id)
    }
}
