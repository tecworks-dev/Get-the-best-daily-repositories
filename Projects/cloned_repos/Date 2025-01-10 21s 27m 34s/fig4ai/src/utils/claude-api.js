import Anthropic from '@anthropic-ai/sdk';

export class ClaudeClient {
    constructor(apiKey) {
        this.client = new Anthropic({
            apiKey: apiKey
        });
    }

    async chat(messages, functions, functionCall) {
        try {
            const systemPrompt = functions ? 
                `You are a function calling AI. Available functions: ${JSON.stringify(functions)}. 
                When responding, you must call one of these functions using the exact format:
                {"name": "function_name", "arguments": {arg1: value1, arg2: value2}}` : undefined;

            const response = await this.client.messages.create({
                model: 'claude-3-sonnet-20240229',
                max_tokens: 4096,
                temperature: 0.7,
                system: systemPrompt,
                messages: messages.map(msg => ({
                    role: msg.role === 'user' ? 'user' : 'assistant',
                    content: msg.content
                }))
            });

            if (functions) {
                // Parse function call from the response content
                try {
                    const text = response.content[0].text;
                    // Find the first JSON object in the response
                    const match = text.match(/\{(?:[^{}]|{[^{}]*})*\}/);
                    if (match) {
                        const parsedCall = JSON.parse(match[0]);
                        if (parsedCall.name && parsedCall.arguments) {
                            return {
                                choices: [{
                                    message: {
                                        function_call: {
                                            name: parsedCall.name,
                                            arguments: JSON.stringify(parsedCall.arguments)
                                        }
                                    }
                                }]
                            };
                        }
                    }
                    // If no valid function call found, throw an error
                    throw new Error('No valid function call found in response');
                } catch (error) {
                    console.error('Error parsing function call from Claude response:', error);
                    throw new Error('Failed to parse function call from response');
                }
            }

            return {
                choices: [{
                    message: {
                        content: response.content[0].text
                    }
                }]
            };
        } catch (error) {
            if (error.message === 'Failed to parse function call from response') {
                throw error;
            }
            throw new Error(`Claude API error: ${error.message}`);
        }
    }
} 