<div align="center">
  <img src="https://raw.githubusercontent.com/jae-jae/fetcher-mcp/refs/heads/main/icon.svg" width="100" height="100" alt="Fetcher MCP Icon" />
</div>

# Fetcher MCP

MCP server for fetch web page content using Playwright headless browser.

## Advantages

- **JavaScript Support**: Unlike traditional web scrapers, Fetcher MCP uses Playwright to execute JavaScript, making it capable of handling dynamic web content and modern web applications.

- **Intelligent Content Extraction**: Built-in Readability algorithm automatically extracts the main content from web pages, removing ads, navigation, and other non-essential elements.

- **Flexible Output Format**: Supports both HTML and Markdown output formats, making it easy to integrate with various downstream applications.

- **Parallel Processing**: The `fetch_urls` tool enables concurrent fetching of multiple URLs, significantly improving efficiency for batch operations.

- **Resource Optimization**: Automatically blocks unnecessary resources (images, stylesheets, fonts, media) to reduce bandwidth usage and improve performance.

- **Robust Error Handling**: Comprehensive error handling and logging ensure reliable operation even when dealing with problematic web pages.

- **Configurable Parameters**: Fine-grained control over timeouts, content extraction, and output formatting to suit different use cases.

## Quick Start

Run directly with npx:

```bash
npx -y fetcher-mcp
```

### Debug Mode

Run with the `--debug` option to show the browser window for debugging:

```bash
npx -y fetcher-mcp --debug
```

## Features

- `fetch_url` - Retrieve web page content from a specified URL
  - Uses Playwright headless browser to parse JavaScript
  - Supports intelligent extraction of main content and conversion to Markdown
  - Supports the following parameters:
    - `url`: The URL of the web page to fetch (required parameter)
    - `timeout`: Page loading timeout in milliseconds, default is 30000 (30 seconds)
    - `waitUntil`: Specifies when navigation is considered complete, options: 'load', 'domcontentloaded', 'networkidle', 'commit', default is 'load'
    - `extractContent`: Whether to intelligently extract the main content, default is true
    - `maxLength`: Maximum length of returned content (in characters), default is no limit
    - `returnHtml`: Whether to return HTML content instead of Markdown, default is false

- `fetch_urls` - Batch retrieve web page content from multiple URLs in parallel
  - Uses multi-tab parallel fetching for improved performance
  - Returns combined results with clear separation between webpages
  - Supports the following parameters:
    - `urls`: Array of URLs to fetch (required parameter)
    - `timeout`: Page loading timeout in milliseconds, default is 30000 (30 seconds)
    - `waitUntil`: Specifies when navigation is considered complete, options: 'load', 'domcontentloaded', 'networkidle', 'commit', default is 'load'
    - `extractContent`: Whether to intelligently extract the main content, default is true
    - `maxLength`: Maximum length of returned content (in characters), default is no limit
    - `returnHtml`: Whether to return HTML content instead of Markdown, default is false

## Configuration MCP

Configure this MCP server in Claude Desktop:

On MacOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

On Windows: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "fetch": {
      "command": "npx",
      "args": ["-y", "fetcher-mcp"]
    }
  }
}
```

## Development

### Install Dependencies

```bash
npm install
```

### Install Playwright Browser

Install the browsers needed for Playwright:

```bash
npm run install-browser
```

### Build the Server

```bash
npm run build
```

## Debugging

Use MCP Inspector for debugging:

```bash
npm run inspector
```

You can also enable visible browser mode for debugging:

```bash
node build/index.js --debug
```

## License

Licensed under the [MIT License](https://choosealicense.com/licenses/mit/)
