import fetch from 'node-fetch';

export async function getFigmaFileData(fileId) {
    const response = await fetch(`https://api.figma.com/v1/files/${fileId}`, {
        headers: {
            'X-Figma-Token': process.env.FIGMA_ACCESS_TOKEN
        }
    });

    if (!response.ok) {
        throw new Error(`Figma API error: ${response.statusText}`);
    }

    return response.json();
} 