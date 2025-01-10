export function processCanvases(document) {
    if (!document || !document.children) return [];

    return document.children.map(canvas => {
        const frames = canvas.children
            ?.filter(child => child.type === 'FRAME')
            ?.map(frame => ({
                id: frame.id,
                name: frame.name,
                type: frame.type,
                size: {
                    width: frame.absoluteBoundingBox?.width || null,
                    height: frame.absoluteBoundingBox?.height || null
                },
                position: {
                    x: frame.x || 0,
                    y: frame.y || 0
                },
                background: frame.backgroundColor,
                layoutMode: frame.layoutMode,
                itemSpacing: frame.itemSpacing,
                padding: {
                    top: frame.paddingTop,
                    right: frame.paddingRight,
                    bottom: frame.paddingBottom,
                    left: frame.paddingLeft
                },
                constraints: frame.constraints,
                clipsContent: frame.clipsContent,
                elements: frame.children?.length || 0
            })) || [];

        return {
            id: canvas.id,
            name: canvas.name,
            type: canvas.type,
            backgroundColor: canvas.backgroundColor,
            children: canvas.children ? canvas.children.length : 0,
            size: {
                width: canvas.absoluteBoundingBox?.width || null,
                height: canvas.absoluteBoundingBox?.height || null
            },
            constraints: canvas.constraints || null,
            exportSettings: canvas.exportSettings || [],
            flowStartingPoints: canvas.flowStartingPoints || [],
            prototypeStartNode: canvas.prototypeStartNode || null,
            frames
        };
    });
}

export function processComponentInstances(node, instances = [], parentName = '') {
    if (!node) return instances;

    const fullName = parentName ? `${parentName}/${node.name}` : node.name;

    if (node.type === 'INSTANCE') {
        instances.push({
            id: node.id,
            name: fullName,
            componentId: node.componentId,
            mainComponent: node.mainComponent,
            styles: node.styles || null,
            position: {
                x: node.x || 0,
                y: node.y || 0
            },
            size: {
                width: node.absoluteBoundingBox?.width || null,
                height: node.absoluteBoundingBox?.height || null
            }
        });
    }

    if (node.children) {
        node.children.forEach(child => {
            processComponentInstances(child, instances, fullName);
        });
    }

    return instances;
}

export function generateComponentYAML(components, instances) {
    // Create a map of component IDs to their instances
    const componentMap = new Map();
    components.forEach(comp => {
        componentMap.set(comp.id, {
            name: comp.name,
            type: comp.type,
            description: comp.description,
            instances: []
        });
    });

    // Map instances to their components
    instances.forEach(instance => {
        if (componentMap.has(instance.componentId)) {
            componentMap.get(instance.componentId).instances.push({
                id: instance.id,
                name: instance.name
            });
        }
    });

    // Generate YAML-like string
    let yaml = 'components:\n';
    componentMap.forEach((value, key) => {
        yaml += `  ${key}:\n`;
        yaml += `    name: "${value.name}"\n`;
        yaml += `    type: ${value.type}\n`;
        if (value.description) {
            yaml += `    description: "${value.description}"\n`;
        }
        if (value.instances.length > 0) {
            yaml += '    instances:\n';
            value.instances.forEach(instance => {
                yaml += `      - id: ${instance.id}\n`;
                yaml += `        name: "${instance.name}"\n`;
            });
        }
        yaml += '\n';
    });

    return yaml;
} 