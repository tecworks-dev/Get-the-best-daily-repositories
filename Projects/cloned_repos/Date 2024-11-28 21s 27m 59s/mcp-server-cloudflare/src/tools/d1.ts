// Add D1 tool definitions
import { config, log } from '../utils/helpers'
import { fetch } from 'undici'
import { Tool } from '@modelcontextprotocol/sdk/types.js'
import { ToolHandlers } from '../utils/types'

const D1_LIST_DATABASES_TOOL: Tool = {
  name: 'd1_list_databases',
  description: 'List all D1 databases in your account',
  inputSchema: {
    type: 'object',
    properties: {},
  },
}
const D1_CREATE_DATABASE_TOOL: Tool = {
  name: 'd1_create_database',
  description: 'Create a new D1 database',
  inputSchema: {
    type: 'object',
    properties: {
      name: {
        type: 'string',
        description: 'Name of the database to create',
      },
    },
    required: ['name'],
  },
}
const D1_DELETE_DATABASE_TOOL: Tool = {
  name: 'd1_delete_database',
  description: 'Delete a D1 database',
  inputSchema: {
    type: 'object',
    properties: {
      databaseId: {
        type: 'string',
        description: 'ID of the database to delete',
      },
    },
    required: ['databaseId'],
  },
}
const D1_QUERY_TOOL: Tool = {
  name: 'd1_query',
  description: 'Execute a SQL query against a D1 database',
  inputSchema: {
    type: 'object',
    properties: {
      databaseId: {
        type: 'string',
        description: 'ID of the database to query',
      },
      query: {
        type: 'string',
        description: 'SQL query to execute',
      },
      params: {
        type: 'array',
        description: 'Optional array of parameters for prepared statements',
        items: {
          type: 'string',
        },
      },
    },
    required: ['databaseId', 'query'],
  },
}
// Add D1 tools to ALL_TOOLS
export const D1_TOOLS = [D1_LIST_DATABASES_TOOL, D1_CREATE_DATABASE_TOOL, D1_DELETE_DATABASE_TOOL, D1_QUERY_TOOL]

// Add D1 response interfaces
interface CloudflareD1DatabasesResponse {
  result: Array<{
    uuid: string
    name: string
    version: string
    created_at: string
    updated_at: string
  }>
  success: boolean
  errors: any[]
  messages: any[]
}

interface CloudflareD1QueryResponse {
  result: Array<any>
  success: boolean
  errors?: any[]
  messages?: any[]
  meta?: {
    changed_db: boolean
    changes?: number
    duration: number
    last_row_id?: number
    rows_read?: number
    rows_written?: number
  }
}

// Add D1 handlers
export async function handleD1ListDatabases() {
  log('Executing d1_list_databases')
  const url = `https://api.cloudflare.com/client/v4/accounts/${config.accountId}/d1/database`

  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${config.apiToken}` },
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`Failed to list D1 databases: ${error}`)
  }

  const data = (await response.json()) as CloudflareD1DatabasesResponse
  return data.result
}

export async function handleD1CreateDatabase(name: string) {
  log('Executing d1_create_database:', name)
  const url = `https://api.cloudflare.com/client/v4/accounts/${config.accountId}/d1/database`

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${config.apiToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ name }),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`Failed to create D1 database: ${error}`)
  }

  const data = (await response.json()) as CloudflareD1DatabasesResponse
  return data.result
}

export async function handleD1DeleteDatabase(databaseId: string) {
  log('Executing d1_delete_database:', databaseId)
  const url = `https://api.cloudflare.com/client/v4/accounts/${config.accountId}/d1/database/${databaseId}`

  const response = await fetch(url, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${config.apiToken}` },
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`Failed to delete D1 database: ${error}`)
  }

  return 'Success'
}

export async function handleD1Query(databaseId: string, query: string, params?: string[]) {
  log('Executing d1_query for database:', databaseId)
  const url = `https://api.cloudflare.com/client/v4/accounts/${config.accountId}/d1/database/${databaseId}/query`

  const body = {
    sql: query,
    params: params || [],
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${config.apiToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`Failed to execute D1 query: ${error}`)
  }

  const data = (await response.json()) as CloudflareD1QueryResponse
  return {
    result: data.result,
    meta: data.meta,
  }
}

export const D1_HANDLERS: ToolHandlers = {
  // Add D1 cases to the tool call handler
  d1_list_databases: async (request) => {
    const results = await handleD1ListDatabases()
    return {
      toolResult: {
        content: [{ type: 'text', text: JSON.stringify(results, null, 2) }],
      },
    }
  },

  d1_create_database: async (request) => {
    const { name } = request.params.arguments as { name: string }
    const result = await handleD1CreateDatabase(name)
    return {
      toolResult: {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
      },
    }
  },

  d1_delete_database: async (request) => {
    const { databaseId } = request.params.arguments as { databaseId: string }
    await handleD1DeleteDatabase(databaseId)
    return {
      toolResult: {
        content: [{ type: 'text', text: `Successfully deleted database: ${databaseId}` }],
      },
    }
  },

  d1_query: async (request) => {
    const { databaseId, query, params } = request.params.arguments as {
      databaseId: string
      query: string
      params?: string[]
    }
    const result = await handleD1Query(databaseId, query, params)
    return {
      toolResult: {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
      },
    }
  },
}
