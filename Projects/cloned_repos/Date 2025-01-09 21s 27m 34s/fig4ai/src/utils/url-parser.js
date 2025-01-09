export function parseFigmaUrl(url) {
    try {
        // Handle URLs without protocol
        const urlWithProtocol = url.startsWith('http') ? url : `https://${url}`;
        const urlObj = new URL(urlWithProtocol);
        
        if (!urlObj.hostname.includes('figma.com')) {
            throw new Error('Not a valid Figma URL');
        }

        const pathParts = urlObj.pathname.split('/').filter(Boolean);
        const fileId = pathParts[1];
        const nodeId = urlObj.searchParams.get('node-id');
        
        // Extract additional parameters
        const page = urlObj.searchParams.get('p');
        const type = urlObj.searchParams.get('t');
        const title = pathParts[2] ? decodeURIComponent(pathParts[2]) : null;

        return {
            type: pathParts[0], // 'file' or 'design'
            fileId,
            nodeId,
            page,
            viewType: type,
            title,
            fullPath: urlObj.pathname,
            originalUrl: url,
            params: Object.fromEntries(urlObj.searchParams)
        };
    } catch (error) {
        throw new Error('Invalid URL format');
    }
} 