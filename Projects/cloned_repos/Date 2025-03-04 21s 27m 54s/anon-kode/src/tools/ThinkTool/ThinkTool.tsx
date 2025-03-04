import { z } from 'zod'
import React from 'react'
import { Text } from 'ink'
import { Tool } from '../../Tool'
import { DESCRIPTION, PROMPT } from './prompt'
import { getTheme } from '../../utils/theme'
import { MessageResponse } from '../../components/MessageResponse'
import { checkGate, logEvent } from '../../services/statsig'
import { USE_BEDROCK, USE_VERTEX } from '../../utils/model'

const thinkToolSchema = z.object({
  thought: z.string().describe('Your thoughts.'),
})

export const ThinkTool = {
  name: 'Think',
  userFacingName: () => 'Think',
  description: async () => DESCRIPTION,
  inputSchema: thinkToolSchema,
  isEnabled: async () =>
    Boolean(process.env.THINK_TOOL) && (await checkGate('tengu_think_tool')),
  isReadOnly: () => true,
  needsPermissions: () => false,
  prompt: async () => PROMPT,

  async *call(input, { messageId }) {
    logEvent('tengu_thinking', {
      messageId,
      thoughtLength: input.thought.length.toString(),
      method: 'tool',
      provider: USE_BEDROCK ? 'bedrock' : USE_VERTEX ? 'vertex' : '1p',
    })

    yield {
      type: 'result',
      resultForAssistant: 'Your thought has been logged.',
      data: { thought: input.thought },
    }
  },

  // This is never called -- it's special-cased in AssistantToolUseMessage
  renderToolUseMessage(input) {
    return input.thought
  },

  renderToolUseRejectedMessage() {
    return (
      <MessageResponse>
        <Text color={getTheme().error}>Thought cancelled</Text>
      </MessageResponse>
    )
  },

  renderResultForAssistant: () => 'Your thought has been logged.',
} satisfies Tool<typeof thinkToolSchema>
