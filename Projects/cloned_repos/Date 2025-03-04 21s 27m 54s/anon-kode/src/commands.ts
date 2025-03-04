import bug from './commands/bug'
import clear from './commands/clear'
import compact from './commands/compact'
import config from './commands/config'
import cost from './commands/cost'
import ctx_viz from './commands/ctx_viz'
import doctor from './commands/doctor'
import help from './commands/help'
import init from './commands/init'
import listen from './commands/listen'
import login from './commands/login'
import logout from './commands/logout'
import * as model from './commands/model'
import onboarding from './commands/onboarding'
import pr_comments from './commands/pr_comments'
import releaseNotes from './commands/release-notes'
import review from './commands/review'
import terminalSetup from './commands/terminalSetup'
import { Tool, ToolUseContext } from './Tool'
import resume from './commands/resume'
import { getMCPCommands } from './services/mcpClient'
import type { MessageParam } from '@anthropic-ai/sdk/resources/index.mjs'
import { memoize } from 'lodash-es'
import type { Message } from './query'
import { isAnthropicAuthEnabled } from './utils/auth'

type PromptCommand = {
  type: 'prompt'
  progressMessage: string
  argNames?: string[]
  getPromptForCommand(args: string): Promise<MessageParam[]>
}

type LocalCommand = {
  type: 'local'
  call(
    args: string,
    context: {
      options: {
        commands: Command[]
        tools: Tool[]
        slowAndCapableModel: string
      }
      abortController: AbortController
      setForkConvoWithMessagesOnTheNextRender: (
        forkConvoWithMessages: Message[],
      ) => void
    },
  ): Promise<string>
}

type LocalJSXCommand = {
  type: 'local-jsx'
  call(
    onDone: (result?: string) => void,
    context: ToolUseContext & {
      setForkConvoWithMessagesOnTheNextRender: (
        forkConvoWithMessages: Message[],
      ) => void
    },
  ): Promise<React.ReactNode>
}

export type Command = {
  description: string
  isEnabled: boolean
  isHidden: boolean
  name: string
  aliases?: string[]
  userFacingName(): string
} & (PromptCommand | LocalCommand | LocalJSXCommand)

const INTERNAL_ONLY_COMMANDS = [ctx_viz, resume, listen]

// Declared as a function so that we don't run this until getCommands is called,
// since underlying functions read from config, which can't be read at module initialization time
const COMMANDS = memoize((): Command[] => [
  clear,
  compact,
  config,
  cost,
  doctor,
  help,
  init,
  model,
  onboarding,
  pr_comments,
  releaseNotes,
  bug,
  review,
  terminalSetup,
  ...(isAnthropicAuthEnabled() ? [logout, login()] : []),
  ...INTERNAL_ONLY_COMMANDS,
])

export const getCommands = memoize(async (): Promise<Command[]> => {
  return [...(await getMCPCommands()), ...COMMANDS()].filter(_ => _.isEnabled)
})

export function hasCommand(commandName: string, commands: Command[]): boolean {
  return commands.some(
    _ => _.userFacingName() === commandName || _.aliases?.includes(commandName),
  )
}

export function getCommand(commandName: string, commands: Command[]): Command {
  const command = commands.find(
    _ => _.userFacingName() === commandName || _.aliases?.includes(commandName),
  ) as Command | undefined
  if (!command) {
    throw ReferenceError(
      `Command ${commandName} not found. Available commands: ${commands
        .map(_ => {
          const name = _.userFacingName()
          return _.aliases ? `${name} (aliases: ${_.aliases.join(', ')})` : name
        })
        .join(', ')}`,
    )
  }

  return command
}
