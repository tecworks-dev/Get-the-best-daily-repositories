import React, { useState, useEffect } from 'react'
import { Box, Text, useInput } from 'ink'
import { getTheme } from '../utils/theme'
import { Select } from './CustomSelect/select'
import { Newline } from 'ink'
import { PRODUCT_NAME } from '../constants/product'
import { useExitOnCtrlCD } from '../hooks/useExitOnCtrlCD'
import {
  getGlobalConfig,
  saveGlobalConfig,
  ProviderType,
} from '../utils/config.js'
import models, { providers } from '../constants/models'
import TextInput from './TextInput'
import OpenAI from 'openai'

type Props = {
  onDone: () => void
}

type ModelInfo = {
  model: string
  provider: string
  [key: string]: any
}

// Define model type options
type ModelTypeOption = 'both' | 'large' | 'small';

export function ModelSelector({ onDone }: Props): React.ReactNode {
  const config = getGlobalConfig()
  const theme = getTheme()
  const exitState = useExitOnCtrlCD(() => process.exit(0))
  
  // State for model configuration
  const [selectedProvider, setSelectedProvider] = useState<ProviderType>(
    config.primaryProvider ?? 'anthropic'
  )
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [apiKey, setApiKey] = useState<string>('')
  
  // UI state
  const [currentScreen, setCurrentScreen] = useState<'modelType' | 'provider' | 'apiKey' | 'model' | 'confirmation'>('modelType')
  const [modelTypeToChange, setModelTypeToChange] = useState<ModelTypeOption>('both')
  
  // Search and model loading state
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([])
  const [isLoadingModels, setIsLoadingModels] = useState(false)
  const [modelLoadError, setModelLoadError] = useState<string | null>(null)
  const [modelSearchQuery, setModelSearchQuery] = useState<string>('')
  const [modelSearchCursorOffset, setModelSearchCursorOffset] = useState<number>(0)
  const [cursorOffset, setCursorOffset] = useState<number>(0)
  // Model type options
  const modelTypeOptions = [
    { label: 'Both Large and Small Models', value: 'both' },
    { label: 'Large Model Only', value: 'large' },
    { label: 'Small Model Only', value: 'small' }
  ]
  
  // Get available providers from models.ts
  const availableProviders = Object.keys(providers)
  
  // Create provider options with nice labels
  const providerOptions = availableProviders.map(provider => {
    const modelCount = models[provider]?.length || 0
    const label = getProviderLabel(provider, modelCount)
    return { 
      label,
      value: provider 
    }
  })
  

  // Create a set of model names from our constants/models.ts for the current provider
  const ourModelNames = new Set(
    (models[selectedProvider as keyof typeof models] || [])
      .map((model: any) => model.model)
  )

  // Create model options from available models, filtered by search query
  const filteredModels = modelSearchQuery 
    ? availableModels.filter(model => 
        model.model.toLowerCase().includes(modelSearchQuery.toLowerCase()))
    : availableModels

  const modelOptions = filteredModels.map(model => {
    // Check if this model is in our constants/models.ts list
    const isInOurModels = ourModelNames.has(model.model)
    
    return {
      label: `${isInOurModels ? '★ ' : ''}${model.model}${getModelDetails(model)}`,
      value: model.model,
      // Add a color property for highlighting
      color: isInOurModels ? theme.suggestion : undefined
    }
  })

  function getModelDetails(model: ModelInfo): string {
    const details = []
    
    if (model.max_tokens) {
      details.push(`${formatNumber(model.max_tokens)} tokens`)
    }
    
    if (model.supports_vision) {
      details.push('vision')
    }
    
    if (model.supports_function_calling) {
      details.push('tools')
    }
    
    return details.length > 0 ? ` (${details.join(', ')})` : ''
  }
  
  function formatNumber(num: number): string {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(1)}M`
    } else if (num >= 1000) {
      return `${(num / 1000).toFixed(0)}K`
    }
    return num.toString()
  }

  function getProviderLabel(provider: string, modelCount: number): string {
    // Use provider names from the providers object if available
    if (providers[provider]) {
      return `${providers[provider].name} ${providers[provider].status === 'wip' ? '(WIP)' : ''} (${modelCount} models)`
    }
    return `${provider}`
  }
  
  function handleModelTypeSelection(type: string) {
    setModelTypeToChange(type as ModelTypeOption)
    setCurrentScreen('provider')
  }

  function handleProviderSelection(provider: string) {
    const providerType = provider as ProviderType
    setSelectedProvider(providerType)
    
    if (provider === 'custom') {
      // For custom provider, save and exit
      saveConfiguration(providerType, selectedModel || config.largeModelName || '')
      onDone()
    } else {
      // For other providers, go to API key input
      setCurrentScreen('apiKey')
    }
  }

  async function fetchGeminiModels() {
    setIsLoadingModels(true)
    setModelLoadError(null)
    
    try {
      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`)

      const { models } = await response.json()

      const geminiModels = models.filter((model: any) => model.supportedGenerationMethods.includes('generateContent')).map((model: any) => ({
        model: model.name.replace('models/', ''),
        provider: 'gemini',
        max_tokens: model.outputTokenLimit,
        supports_vision: model.supports_vision,
        supports_function_calling: model.supports_function_calling
      }))

      setAvailableModels(geminiModels)
      setCurrentScreen('model')
    } catch (error) {
      setModelLoadError(error instanceof Error ? error.message : 'Unknown error')
    } finally {
      setIsLoadingModels(false)
    }
  }
  async function fetchModels() {
    setIsLoadingModels(true)
    setModelLoadError(null)
    
    try {
      const baseURL = providers[selectedProvider]?.baseURL

      const openai = new OpenAI({
        apiKey: apiKey,
        baseURL: baseURL,
        dangerouslyAllowBrowser: true
      })
      
      // Fetch the models
      const response = await openai.models.list()
      
      // Transform the response into our ModelInfo format
      const fetchedModels = [] 
      for (const model of response.data) {
        const modelInfo = models[selectedProvider as keyof typeof models]?.find(m => m.model === model.id)
        fetchedModels.push({
          model: model.id,
          provider: 'openai',
          max_tokens: modelInfo?.max_tokens || 8000,
          supports_vision: modelInfo?.supports_vision || false,
          supports_function_calling: modelInfo?.supports_function_calling || false
        })
      }
      
      setAvailableModels(fetchedModels)
      
      // After fetching models, show the model selection screen
      setCurrentScreen('model')
      // Reset search query when changing providers
      setModelSearchQuery('')
    } catch (error) {
      setModelLoadError(error instanceof Error ? error.message : 'Unknown error')
    } finally {
      setIsLoadingModels(false)
    }
  }
  
  function handleApiKeySubmit(key: string) {
    setApiKey(key)
    
    // Fetch models with the provided API key
    if (selectedProvider === 'gemini') {
      fetchGeminiModels()
    } else {
      fetchModels()
    }
  }
  
  function handleModelSelection(model: string) {
    setSelectedModel(model)
    
    // Show confirmation screen instead of immediately saving
    setCurrentScreen('confirmation')
  }
  
  function saveConfiguration(provider: ProviderType, model: string) {
    const baseURL = providers[provider]?.baseURL || ""
    
    // Create a new config object based on the existing one
    const newConfig = { ...config }
    
    // Update the primary provider regardless of which model we're changing
    newConfig.primaryProvider = provider
    
    // Update the appropriate model based on the selection
    if (modelTypeToChange === 'both' || modelTypeToChange === 'large') {
      newConfig.largeModelName = model
      newConfig.largeModelBaseURL = baseURL
      newConfig.largeModelApiKey = apiKey || config.largeModelApiKey
    }
    
    if (modelTypeToChange === 'both' || modelTypeToChange === 'small') {
      newConfig.smallModelName = model
      newConfig.smallModelBaseURL = baseURL
      newConfig.smallModelApiKey = apiKey || config.smallModelApiKey
    }
    
    // Save the updated configuration
    saveGlobalConfig(newConfig)
  }
  
  function handleConfirmation() {
    // Save the configuration and exit
    saveConfiguration(selectedProvider, selectedModel)
    onDone()
  }
  
  function handleBack() {
    if (currentScreen === 'confirmation') {
      setCurrentScreen('model')
    } else if (currentScreen === 'model') {
      setCurrentScreen('apiKey')
    } else if (currentScreen === 'apiKey') {
      setCurrentScreen('provider')
    } else if (currentScreen === 'provider') {
      setCurrentScreen('modelType')
    }
  }
  
  function handlePastedApiKey(text: string) {
    // Clean up the pasted text (remove whitespace, etc.)
    const cleanedKey = text.trim()
    if (cleanedKey) {
      setApiKey(cleanedKey)
      
      // Update cursor position to end of text
      setCursorOffset(cleanedKey.length)
      
      // Optionally auto-submit if it looks like a valid API key
      if (cleanedKey.startsWith('sk-') && cleanedKey.length > 20) {
        handleApiKeySubmit(cleanedKey)
      }
    }
  }
  
  // Handle cursor offset changes
  function handleCursorOffsetChange(offset: number) {
    setCursorOffset(offset)
  }
  
  // Handle API key changes
  function handleApiKeyChange(value: string) {
    setApiKey(value)
    
    // Update cursor position to end of text when typing
    setCursorOffset(value.length)
  }
  
  // Handle model search query changes
  function handleModelSearchChange(value: string) {
    setModelSearchQuery(value)
    // Update cursor position to end of text when typing
    setModelSearchCursorOffset(value.length)
  }
  
  // Handle model search cursor offset changes
  function handleModelSearchCursorOffsetChange(offset: number) {
    setModelSearchCursorOffset(offset)
  }
  
  // Handle keyboard input
  useInput((input, key) => {
    if (key.escape) {
      handleBack()
    }
    
    if (currentScreen === 'apiKey' && key.return) {
      // Submit API key on Enter
      if (apiKey) {
        handleApiKeySubmit(apiKey)
      }
    }
    
    if (currentScreen === 'confirmation' && key.return) {
      // Confirm selection on Enter
      handleConfirmation()
    }
    
    // Handle paste event (Ctrl+V or Cmd+V)
    if (currentScreen === 'apiKey' && ((key.ctrl && input === 'v') || (key.meta && input === 'v'))) {
      // We can't directly access clipboard in terminal, but we can show a message
      setModelLoadError('Please use your terminal\'s paste functionality or type the API key manually')
    }
  })

  // Render Model Type Selection Screen
  if (currentScreen === 'modelType') {
    return (
      <Box flexDirection="column" gap={1}>
        <Box 
          flexDirection="column" 
          gap={1} 
          borderStyle="round"
          borderColor={theme.secondaryBorder}
          paddingX={2}
          paddingY={1}
        >
          <Text bold>
            {PRODUCT_NAME} Model Configuration {exitState.pending ? `(press ${exitState.keyName} again to exit)` : ''}
          </Text>
          <Box flexDirection="column" gap={1}>
            <Text bold>Which model(s) would you like to configure?</Text>
            <Box flexDirection="column" width={70}>
              <Text color={theme.secondaryText}>
                You can configure both models to be the same, or set them individually.
                <Newline />
                • Large model: Used for complex tasks requiring full capabilities
                <Newline />
                • Small model: Used for simpler tasks to save costs and improve response times
              </Text>
            </Box>
            
            <Select
              options={modelTypeOptions}
              onChange={handleModelTypeSelection}
            />
            
            <Box marginTop={1}>
              <Text dimColor>
                Current configuration:
                <Newline />
                • Large model: <Text color={theme.suggestion}>{config.largeModelName || 'Not set'}</Text>
                {config.largeModelName && (
                  <Text dimColor> ({providers[config.primaryProvider]?.name || config.primaryProvider})</Text>
                )}
                <Newline />
                • Small model: <Text color={theme.suggestion}>{config.smallModelName || 'Not set'}</Text>
                {config.smallModelName && (
                  <Text dimColor> ({providers[config.primaryProvider]?.name || config.primaryProvider})</Text>
                )}
              </Text>
            </Box>
          </Box>
        </Box>
      </Box>
    )
  }

  // Render API Key Input Screen
  if (currentScreen === 'apiKey') {
    const modelTypeText = modelTypeToChange === 'both' 
      ? 'both models' 
      : `your ${modelTypeToChange} model`;
    
    return (
      <Box flexDirection="column" gap={1}>
        <Box 
          flexDirection="column" 
          gap={1} 
          borderStyle="round"
          borderColor={theme.secondaryBorder}
          paddingX={2}
          paddingY={1}
        >
          <Text bold>
            {PRODUCT_NAME} API Key Setup {exitState.pending ? `(press ${exitState.keyName} again to exit)` : ''}
          </Text>
          <Box flexDirection="column" gap={1}>
            <Text bold>Enter your {getProviderLabel(selectedProvider, 0).split(' (')[0]} API key for {modelTypeText}:</Text>
            <Box flexDirection="column" width={70}>
              <Text color={theme.secondaryText}>
                This key will be stored locally and used to access the {selectedProvider} API.
                <Newline />
                Your key is never sent to our servers.
              </Text>
            </Box>
            
            <Box>
              <TextInput
                placeholder="sk-..."
                value={apiKey}
                onChange={handleApiKeyChange}
                onSubmit={handleApiKeySubmit}
                mask="*"
                columns={100}
                cursorOffset={cursorOffset}
                onChangeCursorOffset={handleCursorOffsetChange}
                showCursor={true}
                onPaste={handlePastedApiKey}
              />
            </Box>
            
            <Box marginTop={1}>
              <Text>
                <Text color={theme.suggestion} dimColor={!apiKey}>
                  [Submit API Key]
                </Text>
                <Text> - Press Enter or click to continue with this API key</Text>
              </Text>
            </Box>
            
            {isLoadingModels && (
              <Box>
                <Text color={theme.suggestion}>Loading available models...</Text>
              </Box>
            )}
            {modelLoadError && (
              <Box>
                <Text color="red">Error: {modelLoadError}</Text>
              </Box>
            )}
            <Box marginTop={1}>
              <Text dimColor>
                Press <Text color={theme.suggestion}>Enter</Text> to continue or <Text color={theme.suggestion}>Esc</Text> to go back
              </Text>
            </Box>
          </Box>
        </Box>
      </Box>
    )
  }

  // Render Model Selection Screen
  if (currentScreen === 'model') {
    const modelTypeText = modelTypeToChange === 'both' 
      ? 'both large and small models' 
      : `your ${modelTypeToChange} model`;
    
    return (
      <Box flexDirection="column" gap={1}>
        <Box 
          flexDirection="column" 
          gap={1} 
          borderStyle="round"
          borderColor={theme.secondaryBorder}
          paddingX={2}
          paddingY={1}
        >
          <Text bold>
            {PRODUCT_NAME} Model Selection {exitState.pending ? `(press ${exitState.keyName} again to exit)` : ''}
          </Text>
          <Box flexDirection="column" gap={1}>
            <Text bold>Select a model from {getProviderLabel(selectedProvider, availableModels.length).split(' (')[0]} for {modelTypeText}:</Text>
            <Box flexDirection="column" width={70}>
              <Text color={theme.secondaryText}>
                {modelTypeToChange === 'both' ? (
                  <>This model will be used for both your primary interactions and simpler tasks.</>
                ) : modelTypeToChange === 'large' ? (
                  <>This model will be used for complex tasks requiring full capabilities.</>
                ) : (
                  <>This model will be used for simpler tasks to save costs and improve response times.</>
                )}
                <Newline />
                <Text color={theme.suggestion}>★ Highlighted models</Text> are recommended and well-tested.
              </Text>
            </Box>
            
            <Box marginY={1}>
              <Text bold>Search models:</Text>
              <TextInput
                placeholder="Type to filter models..."
                value={modelSearchQuery}
                onChange={handleModelSearchChange}
                columns={100}
                cursorOffset={modelSearchCursorOffset}
                onChangeCursorOffset={handleModelSearchCursorOffsetChange}
                showCursor={true}
              />
            </Box>
            
            {modelOptions.length > 0 ? (
              <>
                <Select
                  options={modelOptions}
                  onChange={handleModelSelection}
                />
                <Text dimColor>
                  Showing {modelOptions.length} of {availableModels.length} models
                </Text>
              </>
            ) : (
              <Box>
                {availableModels.length > 0 ? (
                  <Text color="yellow">No models match your search. Try a different query.</Text>
                ) : (
                  <Text color="yellow">No models available for this provider.</Text>
                )}
              </Box>
            )}
            
            <Box marginTop={1}>
              <Text dimColor>
                Press <Text color={theme.suggestion}>Esc</Text> to go back to API key input
              </Text>
            </Box>
          </Box>
        </Box>
      </Box>
    )
  }

  // Render Confirmation Screen
  if (currentScreen === 'confirmation') {
    // Determine what will be updated
    const updatingLarge = modelTypeToChange === 'both' || modelTypeToChange === 'large'
    const updatingSmall = modelTypeToChange === 'both' || modelTypeToChange === 'small'
    
    // Get provider display name
    const providerDisplayName = getProviderLabel(selectedProvider, 0).split(' (')[0]
    
    return (
      <Box flexDirection="column" gap={1}>
        <Box 
          flexDirection="column" 
          gap={1} 
          borderStyle="round"
          borderColor={theme.secondaryBorder}
          paddingX={2}
          paddingY={1}
        >
          <Text bold>
            {PRODUCT_NAME} Configuration Confirmation {exitState.pending ? `(press ${exitState.keyName} again to exit)` : ''}
          </Text>
          <Box flexDirection="column" gap={1}>
            <Text bold>Confirm your model configuration:</Text>
            <Box flexDirection="column" width={70}>
              <Text color={theme.secondaryText}>
                Please review your selections before saving.
              </Text>
            </Box>
            
            <Box flexDirection="column" marginY={1} paddingX={1}>
              <Text>
                <Text bold>Provider: </Text>
                <Text color={theme.suggestion}>{providerDisplayName}</Text>
              </Text>
              
              {updatingLarge && (
                <Text>
                  <Text bold>Large Model: </Text>
                  <Text color={theme.suggestion}>{selectedModel}</Text>
                  <Text dimColor> (for complex tasks)</Text>
                </Text>
              )}
              
              {updatingSmall && (
                <Text>
                  <Text bold>Small Model: </Text>
                  <Text color={theme.suggestion}>
                    {modelTypeToChange === 'both' ? selectedModel : config.smallModelName || 'Not set'}
                  </Text>
                  <Text dimColor> (for simpler tasks)</Text>
                </Text>
              )}
              
              {apiKey && (
                <Text>
                  <Text bold>API Key: </Text>
                  <Text color={theme.suggestion}>****{apiKey.slice(-4)}</Text>
                </Text>
              )}
            </Box>
            
            <Box marginTop={1}>
              <Text dimColor>
                Press <Text color={theme.suggestion}>Esc</Text> to go back to model selection or <Text color={theme.suggestion}>Enter</Text> to save configuration
              </Text>
            </Box>
          </Box>
        </Box>
      </Box>
    )
  }

  // Render Provider Selection Screen
  return (
    <Box flexDirection="column" gap={1}>
      <Box 
        flexDirection="column" 
        gap={1} 
        borderStyle="round"
        borderColor={theme.secondaryBorder}
        paddingX={2}
        paddingY={1}
      >
        <Text bold>
          {PRODUCT_NAME} Provider Selection {exitState.pending ? `(press ${exitState.keyName} again to exit)` : ''}
        </Text>
        <Box flexDirection="column" gap={1}>
          <Text bold>
            Select your preferred AI provider for {modelTypeToChange === 'both' 
              ? 'both models' 
              : `your ${modelTypeToChange} model`}:
          </Text>
          <Box flexDirection="column" width={70}>
            <Text color={theme.secondaryText}>
              Choose the provider you want to use for {modelTypeToChange === 'both' 
                ? 'both large and small models' 
                : `your ${modelTypeToChange} model`}.
              <Newline />
              This will determine which models are available to you.
            </Text>
          </Box>
          
          <Select
            options={providerOptions}
            onChange={handleProviderSelection}
          />
          
          <Box marginTop={1}>
            <Text dimColor>
              You can change this later by running <Text color={theme.suggestion}>/model</Text> again
            </Text>
          </Box>
        </Box>
      </Box>
    </Box>
  )
} 