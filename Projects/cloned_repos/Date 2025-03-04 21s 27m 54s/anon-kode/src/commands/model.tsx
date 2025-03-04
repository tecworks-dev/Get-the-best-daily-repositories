import React from 'react'
import { render } from 'ink'
import { ModelSelector } from '../components/ModelSelector'
import { enableConfigs } from '../utils/config'

export const help = 'Change your AI provider and model settings'
export const description = 'Change your AI provider and model settings'
export const isEnabled = true
export const isHidden = false
export const name = 'model'
export const type = 'local-jsx'

export function userFacingName(): string {
  return name
}

export async function call(
  onDone: (result?: string) => void,
): Promise<React.ReactNode> {
  enableConfigs()
  
  return (
    <ModelSelector
      onDone={() => {
        onDone()
      }}
    />
  )
} 