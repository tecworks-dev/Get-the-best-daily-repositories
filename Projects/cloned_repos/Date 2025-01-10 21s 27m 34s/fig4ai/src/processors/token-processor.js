import { rgbToHex } from '../utils/color.js';

export function processDesignTokens(node, tokens = {
    typography: {
        headings: {
            h1: [], h2: [], h3: [], h4: [], h5: [], h6: []
        },
        body: [],
        other: []
    },
    colors: {
        primary: [],
        secondary: [],
        text: [],
        background: [],
        other: []
    },
    spacing: [],
    effects: {
        shadows: [],
        blurs: [],
        other: []
    },
    components: [],
    styles: []
}, parentName = '') {
    if (!node) return tokens;

    const fullName = parentName ? `${parentName}/${node.name}` : node.name;
    const nameLower = node.name.toLowerCase();

    // Process node based on type
    switch (node.type) {
        case 'COMPONENT':
        case 'COMPONENT_SET':
            tokens.components.push({
                id: node.id,
                name: fullName,
                type: node.type,
                description: node.description || null,
                styles: node.styles || null
            });
            break;

        case 'TEXT':
            const textStyle = {
                id: node.id,
                name: fullName,
                content: node.characters,
                style: {
                    fontFamily: node.style?.fontFamily,
                    fontWeight: node.style?.fontWeight,
                    fontSize: node.style?.fontSize,
                    lineHeight: node.style?.lineHeightPx || node.style?.lineHeight,
                    letterSpacing: node.style?.letterSpacing,
                    textCase: node.style?.textCase,
                    textDecoration: node.style?.textDecoration,
                    textAlignHorizontal: node.style?.textAlignHorizontal,
                    paragraphSpacing: node.style?.paragraphSpacing,
                    fills: node.fills
                }
            };

            // Categorize typography
            if (nameLower.includes('heading') || nameLower.match(/h[1-6]/)) {
                const headingLevel = nameLower.match(/h([1-6])/)?.[1];
                if (headingLevel) {
                    tokens.typography.headings[`h${headingLevel}`].push(textStyle);
                }
            } else if (nameLower.includes('body') || nameLower.includes('text') || nameLower.includes('paragraph')) {
                tokens.typography.body.push(textStyle);
            } else {
                tokens.typography.other.push(textStyle);
            }
            break;

        case 'RECTANGLE':
        case 'VECTOR':
        case 'ELLIPSE':
            if (node.fills && node.fills.length > 0) {
                node.fills.forEach(fill => {
                    if (fill.type === 'SOLID') {
                        const colorToken = {
                            id: node.id,
                            name: fullName,
                            color: {
                                r: Math.round(fill.color.r * 255),
                                g: Math.round(fill.color.g * 255),
                                b: Math.round(fill.color.b * 255),
                                a: fill.color.a,
                            },
                            hex: rgbToHex(
                                Math.round(fill.color.r * 255),
                                Math.round(fill.color.g * 255),
                                Math.round(fill.color.b * 255)
                            ),
                            opacity: fill.color.a
                        };

                        // Categorize colors
                        if (nameLower.includes('primary')) {
                            tokens.colors.primary.push(colorToken);
                        } else if (nameLower.includes('secondary')) {
                            tokens.colors.secondary.push(colorToken);
                        } else if (nameLower.includes('text') || nameLower.includes('typography')) {
                            tokens.colors.text.push(colorToken);
                        } else if (nameLower.includes('background') || nameLower.includes('bg')) {
                            tokens.colors.background.push(colorToken);
                        } else {
                            tokens.colors.other.push(colorToken);
                        }
                    }
                });
            }

            // Process effects
            if (node.effects && node.effects.length > 0) {
                node.effects.forEach(effect => {
                    const effectToken = {
                        id: node.id,
                        name: fullName,
                        type: effect.type,
                        value: effect
                    };

                    if (effect.type === 'DROP_SHADOW' || effect.type === 'INNER_SHADOW') {
                        tokens.effects.shadows.push(effectToken);
                    } else if (effect.type === 'LAYER_BLUR' || effect.type === 'BACKGROUND_BLUR') {
                        tokens.effects.blurs.push(effectToken);
                    } else {
                        tokens.effects.other.push(effectToken);
                    }
                });
            }
            break;

        case 'FRAME':
            // Process spacing from auto-layout frames
            if (node.layoutMode === 'VERTICAL' || node.layoutMode === 'HORIZONTAL') {
                tokens.spacing.push({
                    id: node.id,
                    name: fullName,
                    type: node.layoutMode,
                    itemSpacing: node.itemSpacing,
                    padding: {
                        top: node.paddingTop,
                        right: node.paddingRight,
                        bottom: node.paddingBottom,
                        left: node.paddingLeft
                    }
                });
            }
            break;
    }

    // Process styles if present
    if (node.styles) {
        tokens.styles.push({
            id: node.id,
            name: fullName,
            styles: node.styles
        });
    }

    // Recursively process children
    if (node.children) {
        node.children.forEach(child => {
            processDesignTokens(child, tokens, fullName);
        });
    }

    return tokens;
}

export function formatTokenCount(tokens) {
    let counts = {
        typography: Object.values(tokens.typography.headings).flat().length + 
                   tokens.typography.body.length + 
                   tokens.typography.other.length,
        colors: Object.values(tokens.colors).flat().length,
        effects: Object.values(tokens.effects).flat().length,
        spacing: tokens.spacing.length,
        components: tokens.components.length,
        styles: tokens.styles.length
    };
    return Object.entries(counts)
        .map(([key, value]) => `${key}: ${value}`)
        .join(', ');
} 