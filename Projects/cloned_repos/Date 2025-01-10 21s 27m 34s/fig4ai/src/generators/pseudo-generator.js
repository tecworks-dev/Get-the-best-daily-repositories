import OpenAI from 'openai';
import ora from 'ora';
import chalk from 'chalk';
import { rgbToHex } from '../utils/color.js';
import { ClaudeClient } from '../utils/claude-api.js';

let client;
let hasAICapability = false;

export function initializeAI(model = 'claude') {
    // Check if --no-ai flag is present
    if (process.argv.includes('--no-ai')) {
        hasAICapability = false;
        return;
    }

    try {
        if (model === 'gpt4' && process.env.OPENAI_API_KEY) {
            client = new OpenAI({
                apiKey: process.env.OPENAI_API_KEY
            });
            hasAICapability = true;
        } else if (model === 'claude' && process.env.CLAUDE_API_KEY) {
            client = new ClaudeClient(process.env.CLAUDE_API_KEY);
            hasAICapability = true;
        }
    } catch (error) {
        console.warn(chalk.yellow('Failed to initialize AI client:', error.message));
        hasAICapability = false;
    }
}

async function generatePseudoComponent(component, instance, tokens, figmaData) {
    if (!hasAICapability || !client) {
        return {
            componentName: component.name,
            pseudoCode: `# ${component.name}\n\`\`\`\n${JSON.stringify(instance, null, 2)}\n\`\`\``
        };
    }

    // Create a more detailed design system summary with exact values
    const designSystem = {
        typography: {
            headings: Object.fromEntries(
                Object.entries(tokens.typography.headings)
                    .map(([key, styles]) => [key, styles[0]?.style || null])
                    .filter(([_, style]) => style !== null)
            ),
            body: tokens.typography.body[0]?.style || null
        },
        colors: {
            primary: tokens.colors.primary.map(c => ({ 
                name: c.name, 
                hex: c.hex,
                rgb: `${c.color.r},${c.color.g},${c.color.b}`,
                opacity: c.opacity
            })),
            secondary: tokens.colors.secondary.map(c => ({ 
                name: c.name, 
                hex: c.hex,
                rgb: `${c.color.r},${c.color.g},${c.color.b}`,
                opacity: c.opacity
            })),
            text: tokens.colors.text.map(c => ({ 
                name: c.name, 
                hex: c.hex,
                rgb: `${c.color.r},${c.color.g},${c.color.b}`,
                opacity: c.opacity
            })),
            background: tokens.colors.background.map(c => ({ 
                name: c.name, 
                hex: c.hex,
                rgb: `${c.color.r},${c.color.g},${c.color.b}`,
                opacity: c.opacity
            })),
            other: tokens.colors.other.map(c => ({ 
                name: c.name, 
                hex: c.hex,
                rgb: `${c.color.r},${c.color.g},${c.color.b}`,
                opacity: c.opacity
            }))
        },
        spacing: tokens.spacing.map(s => ({
            name: s.name,
            value: s.itemSpacing,
            padding: s.padding
        })),
        effects: {
            shadows: tokens.effects.shadows.map(s => ({
                name: s.name,
                type: s.type,
                ...s.value,
                color: s.value.color ? {
                    hex: rgbToHex(
                        Math.round(s.value.color.r * 255),
                        Math.round(s.value.color.g * 255),
                        Math.round(s.value.color.b * 255)
                    ),
                    rgb: `${Math.round(s.value.color.r * 255)},${Math.round(s.value.color.g * 255)},${Math.round(s.value.color.b * 255)}`,
                    opacity: s.value.color.a
                } : null
            })),
            blurs: tokens.effects.blurs.map(b => ({
                name: b.name,
                type: b.type,
                ...b.value
            }))
        }
    };

    // Extract component-specific styles and references
    const componentStyles = {
        styles: {},  // Will be populated with expanded styles
        fills: instance.fills?.map(fill => {
            if (fill.type === 'SOLID') {
                // Check if this fill comes from a style
                const styleId = instance.styles?.fills || instance.styles?.fill;
                if (styleId) {
                    // Find the style in tokens
                    const style = tokens.styles.find(s => s.id === styleId);
                    // Find the actual style definition in the Figma data
                    const styleDefinition = figmaData.styles?.[styleId];
                    return {
                        type: fill.type,
                        styleId,
                        styleName: style?.name || 'Unknown Style',
                        styleType: 'fill',
                        description: styleDefinition?.description || null,
                        color: {
                            hex: rgbToHex(
                                Math.round(fill.color.r * 255),
                                Math.round(fill.color.g * 255),
                                Math.round(fill.color.b * 255)
                            ),
                            rgb: `${Math.round(fill.color.r * 255)},${Math.round(fill.color.g * 255)},${Math.round(fill.color.b * 255)}`,
                            opacity: fill.color.a
                        }
                    };
                }
                return {
                    type: fill.type,
                    color: {
                        hex: rgbToHex(
                            Math.round(fill.color.r * 255),
                            Math.round(fill.color.g * 255),
                            Math.round(fill.color.b * 255)
                        ),
                        rgb: `${Math.round(fill.color.r * 255)},${Math.round(fill.color.g * 255)},${Math.round(fill.color.b * 255)}`,
                        opacity: fill.color.a
                    }
                };
            }
            return fill;
        }),
        effects: instance.effects?.map(effect => {
            const styleId = instance.styles?.effects || instance.styles?.effect;
            if (styleId) {
                const style = tokens.styles.find(s => s.id === styleId);
                const styleDefinition = figmaData.styles?.[styleId];
                return {
                    type: effect.type,
                    styleId,
                    styleName: style?.name || 'Unknown Style',
                    styleType: 'effect',
                    description: styleDefinition?.description || null,
                    value: {
                        ...effect,
                        color: effect.color ? {
                            hex: rgbToHex(
                                Math.round(effect.color.r * 255),
                                Math.round(effect.color.g * 255),
                                Math.round(effect.color.b * 255)
                            ),
                            rgb: `${Math.round(effect.color.r * 255)},${Math.round(effect.color.g * 255)},${Math.round(effect.color.b * 255)}`,
                            opacity: effect.color.a
                        } : null
                    }
                };
            }
            return effect;
        })
    };

    // Expand all style references
    if (instance.styles) {
        Object.entries(instance.styles).forEach(([key, styleId]) => {
            const style = tokens.styles.find(s => s.id === styleId);
            const styleDefinition = figmaData.styles?.[styleId];
            
            componentStyles.styles[key] = {
                id: styleId,
                name: style?.name || 'Unknown Style',
                type: key,
                description: styleDefinition?.description || null,
                value: styleDefinition || null
            };
        });
    }

    const functions = [
        {
            name: "create_pseudo_component",
            description: "Generate a pseudo-XML component based on Figma component details",
            parameters: {
                type: "object",
                properties: {
                    componentName: {
                        type: "string",
                        description: "The name of the component"
                    },
                    pseudoCode: {
                        type: "string",
                        description: "The pseudo-XML code for the component with detailed styling"
                    }
                },
                required: ["componentName", "pseudoCode"]
            }
        }
    ];

    const prompt = `Design System Details:

\`\`\`
${JSON.stringify(designSystem, null, 2)}
\`\`\`

Component to Generate:
Name: ${component.name}
Type: ${component.type}
Description: ${component.description || 'No description provided'}
Size: ${instance.size.width}x${instance.size.height}

Component Specific Styles and References:
\`\`\`
${JSON.stringify(componentStyles, null, 2)}
\`\`\`

Requirements:
1. Generate pseudo-XML code that represents this component
2. Use style references (styleId) when available instead of direct values
3. Include ALL styling details (colors, shadows, effects)
4. Use exact color values (HEX and RGB) when no style reference exists
5. Include shadow and effect details with style references
6. Specify padding and spacing
7. Include background colors and gradients
8. Make it accessible
9. Keep it readable

Example format:
<Button 
  fills="style_id_123"
  effects="style_id_456"
  strokes="style_id_789"
  padding="8px 16px"
  border-radius="4px"
>
  <Icon name="star" fills="style_id_234" />
  <Text fills="style_id_567" font-size="16px">Click me</Text>
</Button>

Generate ONLY the pseudo-XML code with detailed styling attributes, preferring style references over direct values.`;

    try {
        const completion = await client.chat(
            [{ role: "user", content: prompt }],
            functions,
            { name: "create_pseudo_component" }
        );

        const response = JSON.parse(completion.choices[0].message.function_call.arguments);
        return response;
    } catch (error) {
        console.warn(chalk.yellow(`Skipping pseudo generation for component ${component.name} - ${error.message}`));
        return {
            componentName: component.name,
            pseudoCode: `# ${component.name}\n${JSON.stringify(instance, null, 2)}`
        };
    }
}

async function generatePseudoFrame(frame, components, tokens, canvas) {
    if (!hasAICapability || !client) {
        return {
            frameName: frame.name,
            pseudoCode: `# ${frame.name} (Canvas: ${canvas.name})\n${JSON.stringify(frame, null, 2)}`
        };
    }

    const functions = [
        {
            name: "create_pseudo_frame",
            description: "Generate a pseudo-XML frame layout based on Figma frame details",
            parameters: {
                type: "object",
                properties: {
                    frameName: {
                        type: "string",
                        description: "The name of the frame"
                    },
                    pseudoCode: {
                        type: "string",
                        description: "The pseudo-XML code for the frame layout"
                    }
                },
                required: ["frameName", "pseudoCode"]
            }
        }
    ];

    // Extract frame dimensions and properties for the summary
    const frameSize = frame.absoluteBoundingBox ? {
        width: frame.absoluteBoundingBox.width,
        height: frame.absoluteBoundingBox.height
    } : { width: 0, height: 0 };

    const framePadding = {
        top: frame.paddingTop || 0,
        right: frame.paddingRight || 0,
        bottom: frame.paddingBottom || 0,
        left: frame.paddingLeft || 0
    };

    const canvasSize = canvas.absoluteBoundingBox ? {
        width: canvas.absoluteBoundingBox.width,
        height: canvas.absoluteBoundingBox.height
    } : { width: 0, height: 0 };

    const prompt = `Frame Summary:
Name: ${frame.name}
Size: ${frameSize.width}x${frameSize.height}
Layout: ${frame.layoutMode || 'FREE'}
Spacing: ${frame.itemSpacing || 0}
Padding: ${JSON.stringify(framePadding)}
Elements: ${frame.children?.length || 0}
Position: x=${frame.absoluteBoundingBox?.x || 0}, y=${frame.absoluteBoundingBox?.y || 0}

Canvas Summary:
Name: ${canvas.name}
Type: ${canvas.type}
Size: ${canvasSize.width}x${canvasSize.height}

Available Components:
${components.map(c => `- ${c.name}`).join('\n')}

Complete Frame Data:
\`\`\`
${JSON.stringify(frame, null, 2)}
\`\`\`

Complete Canvas Data:
\`\`\`
${JSON.stringify(canvas, null, 2)}
\`\`\`

Requirements:
1. Generate pseudo-XML layout code for this frame
2. Use semantic container elements
3. Include layout attributes (flex, grid, etc.)
4. Use appropriate spacing and padding
5. Place components in a logical layout
6. Consider canvas context for positioning and constraints
7. Include all text content exactly as specified in the frame data
8. Preserve all styling information from the frame data
9. Keep the hierarchy of nested elements
10. Keep it readable while being accurate to the source data

Example format:
<Frame 
  name="${frame.name}" 
  layout="${frame.layoutMode || 'FREE'}" 
  spacing="${frame.itemSpacing || 0}" 
  canvas="${canvas.name}"
  position="x=${frame.absoluteBoundingBox?.x || 0},y=${frame.absoluteBoundingBox?.y || 0}"
  size="w=${frameSize.width},h=${frameSize.height}"
  constraints="${JSON.stringify(frame.constraints)}"
  background="${JSON.stringify(frame.backgroundColor)}"
  blendMode="${frame.blendMode}"
  clipsContent="${frame.clipsContent}"
>
  <!-- Generate nested elements based on frame.children -->
  <!-- Include all text content, styles, and properties -->
  <!-- Use style references when available -->
</Frame>

Generate ONLY the pseudo-XML code without any additional explanation. Ensure all text content and styling from the frame data is accurately represented.`;

    try {
        const completion = await client.chat(
            [{ role: "user", content: prompt }],
            functions,
            { name: "create_pseudo_frame" }
        );

        const response = JSON.parse(completion.choices[0].message.function_call.arguments);
        return response;
    } catch (error) {
        console.warn(chalk.yellow(`Skipping pseudo generation for frame ${frame.name} - ${error.message}`));
        return {
            frameName: frame.name,
            pseudoCode: `# ${frame.name} (Canvas: ${canvas.name})\n${JSON.stringify(frame, null, 2)}`
        };
    }
}

export async function generateAllPseudoCode(components, instances, frames, tokens, figmaData) {
    const pseudoComponents = new Map();
    const spinner = ora();

    if (!hasAICapability) {
        spinner.info('Running without AI enhancement - will output raw data');
    }

    // Generate components first
    spinner.start('Processing components...');
    for (const component of components) {
        spinner.text = `Processing component: ${component.name}`;
        const componentInstances = instances.filter(i => i.componentId === component.id);
        if (componentInstances.length > 0) {
            const mainInstance = componentInstances[0];
            const pseudoComponent = await generatePseudoComponent(component, mainInstance, tokens, figmaData);
            if (pseudoComponent) {
                pseudoComponents.set(component.id, pseudoComponent);
                spinner.stop();
                console.log(chalk.green(`✓ Processed component: ${component.name}`));
                spinner.start();
            }
        }
    }
    spinner.succeed('All components processed');

    spinner.start('Processing frame layouts...');
    const pseudoFrames = new Map();

    // Generate frames using the components
    for (const canvas of figmaData.document.children) {
        spinner.stop();
        console.log(chalk.blue(`\nProcessing canvas: ${canvas.name}`));
        spinner.start();
        for (const frame of canvas.children?.filter(child => child.type === 'FRAME') || []) {
            spinner.text = `Processing frame: ${frame.name}`;
            const pseudoFrame = await generatePseudoFrame(frame, components, tokens, canvas);
            if (pseudoFrame) {
                pseudoFrames.set(frame.id, pseudoFrame);
                spinner.stop();
                console.log(chalk.green(`  ✓ Processed frame: ${frame.name}`));
                spinner.start();
            }
        }
    }
    spinner.succeed('All frames processed');

    return { components: pseudoComponents, frames: pseudoFrames };
} 