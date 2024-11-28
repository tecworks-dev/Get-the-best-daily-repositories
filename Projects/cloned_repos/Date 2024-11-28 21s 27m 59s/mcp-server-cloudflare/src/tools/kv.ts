import { config, log } from '../utils/helpers'
import { fetch } from 'undici'
import { ToolHandlers } from '../utils/types'
import { Tool } from '@modelcontextprotocol/sdk/types.js'

interface CloudflareListResponse {
  result: Array<{
    name: string
    expiration?: number
  }>
  success: boolean
  errors: any[]
  messages: any[]
}

// Add the new KV list namespaces tool definition
const GET_KVS_TOOL: Tool = {
  name: 'get_kvs',
  description: 'List KV namespaces in your account',
  inputSchema: {
    type: 'object',
    properties: {},
  },
}
// Modify existing KV tool definitions to include namespaceId
const KV_GET_TOOL: Tool = {
  name: 'kv_get',
  description: 'Get a value from Cloudflare KV store',
  inputSchema: {
    type: 'object',
    properties: {
      namespaceId: {
        type: 'string',
        description: 'The KV namespace ID',
      },
      key: {
        type: 'string',
        description: 'The key to retrieve',
      },
    },
    required: ['namespaceId', 'key'],
  },
}
const KV_PUT_TOOL: Tool = {
  name: 'kv_put',
  description: 'Put a value into Cloudflare KV store',
  inputSchema: {
    type: 'object',
    properties: {
      namespaceId: {
        type: 'string',
        description: 'The KV namespace ID',
      },
      key: {
        type: 'string',
        description: 'The key to store',
      },
      value: {
        type: 'string',
        description: 'The value to store',
      },
      expirationTtl: {
        type: 'number',
        description: 'Optional expiration time in seconds',
      },
    },
    required: ['namespaceId', 'key', 'value'],
  },
}
const KV_DELETE_TOOL: Tool = {
  name: 'kv_delete',
  description: 'Delete a key from Cloudflare KV store',
  inputSchema: {
    type: 'object',
    properties: {
      namespaceId: {
        type: 'string',
        description: 'The KV namespace ID',
      },
      key: {
        type: 'string',
        description: 'The key to delete',
      },
    },
    required: ['namespaceId', 'key'],
  },
}
const KV_LIST_TOOL: Tool = {
  name: 'kv_list',
  description: 'List keys in Cloudflare KV store',
  inputSchema: {
    type: 'object',
    properties: {
      namespaceId: {
        type: 'string',
        description: 'The KV namespace ID',
      },
      prefix: {
        type: 'string',
        description: 'Optional prefix to filter keys',
      },
      limit: {
        type: 'number',
        description: 'Maximum number of keys to return',
      },
    },
    required: ['namespaceId'],
  },
}
export const KV_TOOLS = [GET_KVS_TOOL, KV_GET_TOOL, KV_PUT_TOOL, KV_DELETE_TOOL, KV_LIST_TOOL]

// Add interface for KV namespace response
interface CloudflareKVNamespacesResponse {
  result: Array<{
    id: string
    title: string
    supports_url_encoding?: boolean
  }>
  success: boolean
  errors: any[]
  messages: any[]
}

// Add handler for getting KV namespaces
export async function handleGetKVs() {
  log('Executing get_kvs')
  const url = `https://api.cloudflare.com/client/v4/accounts/${config.accountId}/storage/kv/namespaces`

  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${config.apiToken}` },
  })

  log('KV namespaces response status:', response.status)

  if (!response.ok) {
    const error = await response.text()
    log('KV namespaces error:', error)
    throw new Error(`Failed to list KV namespaces: ${error}`)
  }

  const data = (await response.json()) as CloudflareKVNamespacesResponse
  log('KV namespaces success:', data)
  return data.result
}

// Modify existing handlers to accept namespaceId
export async function handleGet(namespaceId: string, key: string) {
  log('Executing kv_get for key:', key, 'in namespace:', namespaceId)
  const url = `https://api.cloudflare.com/client/v4/accounts/${config.accountId}/storage/kv/namespaces/${namespaceId}/values/${key}`

  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${config.apiToken}` },
  })

  log('KV get response status:', response.status)

  if (!response.ok) {
    const error = await response.text()
    log('KV get error:', error)
    throw new Error(`Failed to get value: ${error}`)
  }

  const value = await response.text()
  log('KV get success:', value)
  return value
}

export async function handlePut(namespaceId: string, key: string, value: string, expirationTtl?: number) {
  log('Executing kv_put for key:', key, 'in namespace:', namespaceId)
  const url = `https://api.cloudflare.com/client/v4/accounts/${config.accountId}/storage/kv/namespaces/${namespaceId}/values/${key}`

  const response = await fetch(url, {
    method: 'PUT',
    headers: {
      Authorization: `Bearer ${config.apiToken}`,
      'Content-Type': 'text/plain',
    },
    body: value,
    ...(expirationTtl ? { query: { expiration_ttl: expirationTtl } } : {}),
  })

  log('KV put response status:', response.status)

  if (!response.ok) {
    const error = await response.text()
    log('KV put error:', error)
    throw new Error(`Failed to put value: ${error}`)
  }

  return 'Success'
}

export async function handleDelete(namespaceId: string, key: string) {
  log('Executing kv_delete for key:', key, 'in namespace:', namespaceId)
  const url = `https://api.cloudflare.com/client/v4/accounts/${config.accountId}/storage/kv/namespaces/${namespaceId}/values/${key}`

  const response = await fetch(url, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${config.apiToken}` },
  })

  log('KV delete response status:', response.status)

  if (!response.ok) {
    const error = await response.text()
    log('KV delete error:', error)
    throw new Error(`Failed to delete key: ${error}`)
  }

  return 'Success'
}

export async function handleList(namespaceId: string, prefix?: string, limit?: number) {
  log('Executing kv_list in namespace:', namespaceId)
  const params = new URLSearchParams()
  if (prefix) params.append('prefix', prefix)
  if (limit) params.append('limit', limit.toString())

  const url = `https://api.cloudflare.com/client/v4/accounts/${config.accountId}/storage/kv/namespaces/${namespaceId}/keys?${params}`

  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${config.apiToken}` },
  })

  log('KV list response status:', response.status)

  if (!response.ok) {
    const error = await response.text()
    log('KV list error:', error)
    throw new Error(`Failed to list keys: ${error}`)
  }

  const data = (await response.json()) as CloudflareListResponse
  log('KV list success:', data)
  return data.result
}

export const KV_HANDLERS: ToolHandlers = {
  get_kvs: async (request) => {
    const results = await handleGetKVs()
    return {
      toolResult: {
        content: [
          {
            type: 'text',
            text: JSON.stringify(results, null, 2),
          },
        ],
      },
    }
  },

  kv_get: async (request) => {
    const { namespaceId, key } = request.params.arguments as { namespaceId: string; key: string }
    const value = await handleGet(namespaceId, key)
    return {
      toolResult: {
        content: [
          {
            type: 'text',
            text: value,
          },
        ],
      },
    }
  },

  kv_put: async (request) => {
    const { namespaceId, key, value, expirationTtl } = request.params.arguments as {
      namespaceId: string
      key: string
      value: string
      expirationTtl?: number
    }
    await handlePut(namespaceId, key, value, expirationTtl)
    return {
      toolResult: {
        content: [
          {
            type: 'text',
            text: `Successfully stored value for key: ${key}`,
          },
        ],
      },
    }
  },

  kv_delete: async (request) => {
    const { namespaceId, key } = request.params.arguments as { namespaceId: string; key: string }
    await handleDelete(namespaceId, key)
    return {
      toolResult: {
        content: [
          {
            type: 'text',
            text: `Successfully deleted key: ${key}`,
          },
        ],
      },
    }
  },

  kv_list: async (request) => {
    const { namespaceId, prefix, limit } = request.params.arguments as {
      namespaceId: string
      prefix?: string
      limit?: number
    }
    const results = await handleList(namespaceId, prefix, limit)
    return {
      toolResult: {
        content: [
          {
            type: 'text',
            text: JSON.stringify(results, null, 2),
          },
        ],
      },
    }
  },
}
