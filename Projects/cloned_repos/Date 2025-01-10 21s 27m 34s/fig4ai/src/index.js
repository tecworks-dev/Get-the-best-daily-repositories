#!/usr/bin/env node

// Suppress punycode deprecation warning
process.noDeprecation = true;

import chalk from 'chalk';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';
import ora from 'ora';

import { parseFigmaUrl } from './utils/url-parser.js';
import { getFigmaFileData } from './utils/api.js';
import { processDesignTokens, formatTokenCount } from './processors/token-processor.js';
import { processCanvases, processComponentInstances, generateComponentYAML } from './processors/canvas-processor.js';
import { generateAllPseudoCode, initializeAI } from './generators/pseudo-generator.js';

// Load environment variables
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
dotenv.config({ path: join(dirname(__dirname), '.env') });

// Validate required environment variables
const requiredEnvVars = {
    'FIGMA_ACCESS_TOKEN': process.env.FIGMA_ACCESS_TOKEN
};

// Optional environment variables
const optionalEnvVars = {
    'OPENAI_API_KEY': process.env.OPENAI_API_KEY,
    'CLAUDE_API_KEY': process.env.CLAUDE_API_KEY
};

const missingEnvVars = Object.entries(requiredEnvVars)
    .filter(([_, value]) => !value)
    .map(([key]) => key);

if (missingEnvVars.length > 0) {
    console.error(chalk.red('\nMissing required environment variables:'));
    missingEnvVars.forEach(envVar => {
        console.error(chalk.yellow(`  • ${envVar}`));
    });
    console.error(chalk.blue('\nPlease set these variables in your .env file:'));
    console.error(chalk.gray('\n# .env'));
    missingEnvVars.forEach(envVar => {
        console.error(chalk.gray(`${envVar}=your_${envVar.toLowerCase()}_here`));
    });
    process.exit(1);
}

// Parse command line arguments
const args = process.argv.slice(2);
const figmaUrl = args[0] || process.env.FIGMA_DESIGN_URL;
const modelArg = args.find(arg => arg.startsWith('--model='));
const noAI = args.includes('--no-ai');
const model = modelArg ? modelArg.split('=')[1].toLowerCase() : 'claude';

if (!figmaUrl) {
    console.error(chalk.red('Please provide a Figma URL'));
    console.log(chalk.blue('\nUsage:'));
    console.log('  npx fig4ai <figma-url> [--model=claude|gpt4] [--no-ai]');
    console.log(chalk.blue('\nOptions:'));
    console.log('  --model=claude|gpt4    Choose AI model (default: claude)');
    console.log('  --no-ai                Skip AI enhancements and output raw data');
    console.log(chalk.blue('\nOr set it in your .env file:'));
    console.log(chalk.gray('FIGMA_DESIGN_URL=your_figma_url_here'));
    process.exit(1);
}

// Check if AI enhancement is possible and desired
const hasAICapability = !noAI && ((model === 'claude' && process.env.CLAUDE_API_KEY) || 
                                 (model === 'gpt4' && process.env.OPENAI_API_KEY));

if (noAI) {
    console.info(chalk.blue('\nAI enhancement disabled via --no-ai flag.'));
} else if (!hasAICapability) {
    console.warn(chalk.yellow('\nNo AI API keys found. Running without AI enhancement.'));
    console.warn(chalk.gray('To enable AI features, set CLAUDE_API_KEY or OPENAI_API_KEY in your .env file.'));
}

async function main() {
    const spinner = ora();
    try {
        // Initialize AI with selected model
        initializeAI(model);

        const result = parseFigmaUrl(figmaUrl);
        let output = '';

        // Capture URL details
        output += '# Figma Design Rules\n\n';
        output += '## File Information\n';
        output += `Type: ${result.type}\n`;
        output += `File ID: ${result.fileId}\n`;
        output += `Title: ${result.title || 'Not specified'}\n`;
        output += `Node ID: ${result.nodeId || 'Not specified'}\n\n`;

        spinner.start('Processing Figma URL details...');
        spinner.succeed('Figma URL details processed');

        spinner.start('Fetching file data from Figma API...');
        const figmaData = await getFigmaFileData(result.fileId);
        spinner.succeed('Figma file data fetched');
        
        output += `File Name: ${figmaData.name}\n`;
        output += `Last Modified: ${new Date(figmaData.lastModified).toLocaleString()}\n\n`;

        spinner.start('Processing design tokens...');
        const tokens = processDesignTokens(figmaData.document);
        spinner.succeed('Design tokens processed');
        
        // Add token summary
        output += '## Design Tokens Summary\n';
        output += formatTokenCount(tokens) + '\n\n';

        spinner.info(`Total tokens found: ${formatTokenCount(tokens)}`);

        // Process and capture detailed token information
        spinner.start('Processing typography tokens...');
        output += '## Typography\n\n';
        Object.entries(tokens.typography.headings).forEach(([level, styles]) => {
            if (styles.length > 0) {
                output += `### ${level.toUpperCase()}\n`;
                styles.forEach(style => {
                    output += `- ${style.name}\n`;
                    output += `  - Font: ${style.style.fontFamily} (${style.style.fontWeight})\n`;
                    output += `  - Size: ${style.style.fontSize}px\n`;
                    output += `  - Line Height: ${style.style.lineHeight}\n`;
                    if (style.style.letterSpacing) {
                        output += `  - Letter Spacing: ${style.style.letterSpacing}\n`;
                    }
                    output += '\n';
                });
            }
        });
        spinner.succeed('Typography tokens processed');

        spinner.start('Processing color tokens...');
        output += '## Colors\n\n';
        Object.entries(tokens.colors).forEach(([category, colors]) => {
            if (colors.length > 0) {
                output += `### ${category.toUpperCase()}\n`;
                colors.forEach(color => {
                    output += `- ${color.name}\n`;
                    output += `  - HEX: ${color.hex}\n`;
                    output += `  - RGB: ${color.color.r}, ${color.color.g}, ${color.color.b}\n`;
                    if (color.opacity !== 1) {
                        output += `  - Opacity: ${color.opacity}\n`;
                    }
                    output += '\n';
                });
            }
        });
        spinner.succeed('Color tokens processed');

        // Process canvas information
        spinner.start('Processing canvas information...');
        const canvases = processCanvases(figmaData.document);
        output += '## Canvases and Frames\n\n';
        canvases.forEach(canvas => {
            output += `### ${canvas.name}\n`;
            output += `- ID: ${canvas.id}\n`;
            output += `- Type: ${canvas.type}\n`;
            output += `- Total Elements: ${canvas.children}\n`;
            if (canvas.frames && canvas.frames.length > 0) {
                output += `\n#### Frames (${canvas.frames.length})\n`;
                canvas.frames.forEach(frame => {
                    output += `\n##### ${frame.name}\n`;
                    output += `- ID: ${frame.id}\n`;
                    if (frame.size.width && frame.size.height) {
                        output += `- Size: ${frame.size.width}x${frame.size.height}\n`;
                    }
                    if (frame.layoutMode) {
                        output += `- Layout: ${frame.layoutMode}\n`;
                        output += `- Item Spacing: ${frame.itemSpacing}\n`;
                    }
                });
            }
            output += '\n';
        });
        spinner.succeed('Canvas information processed');

        // Process component instances
        spinner.start('Processing component instances...');
        const instances = processComponentInstances(figmaData.document);
        output += '## Component Instances\n\n';
        instances.forEach(instance => {
            output += `### ${instance.name}\n`;
            output += `- ID: ${instance.id}\n`;
            output += `- Component ID: ${instance.componentId}\n`;
            if (instance.size.width && instance.size.height) {
                output += `- Size: ${instance.size.width}x${instance.size.height}\n`;
            }
            output += '\n';
        });
        spinner.succeed('Component instances processed');

        // Generate component structure
        spinner.start('Generating component structure...');
        output += '## Component Structure\n\n```yaml\n';
        const componentYAML = generateComponentYAML(tokens.components, instances);
        output += componentYAML;
        output += '```\n\n';
        spinner.succeed('Component structure generated');

        // Generate pseudo components and frames
        spinner.start('Generating pseudo components and frames...');
        const frames = canvases.flatMap(canvas => canvas.frames);
        const pseudoCode = await generateAllPseudoCode(tokens.components, instances, frames, tokens, figmaData);
        spinner.succeed('Pseudo components and frames generated');
        
        // Add pseudo code
        output += '## Pseudo Components\n\n```xml\n';
        pseudoCode.components.forEach((component, id) => {
            output += component.pseudoCode + '\n\n';
        });
        output += '```\n\n';

        output += '## Frame Layouts\n\n```xml\n';
        pseudoCode.frames.forEach((frame, id) => {
            output += frame.pseudoCode + '\n\n';
        });
        output += '```\n';

        // Save to .designrules file
        spinner.start('Saving design rules...');
        await fs.promises.writeFile('.designrules', output);
        spinner.succeed('Design rules saved successfully');

    } catch (error) {
        spinner.fail(chalk.red('Error: ' + error.message));
        process.exit(1);
    }
}

main(); 