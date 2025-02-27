import TurndownService from 'turndown';
import render from 'dom-serializer';
import * as cheerio from 'cheerio';
// import { gfm } from 'turndown-plugin-gfm';

const logger = console;

const turndownService = new TurndownService({
  headingStyle: 'atx',
  hr: '---',
  bulletListMarker: '*',
  codeBlockStyle: 'fenced',
  fence: '```',
  emDelimiter: '*', // unlike underscore, this works also intra-word
  strongDelimiter: '**', // unlike underscores, this works also intra-word
  linkStyle: 'inlined',
  linkReferenceStyle: 'full',
  br: '  ',
});

/**
 * Remove all style and script tags
 */
turndownService.addRule('remove', {
  filter: ['style', 'script', 'aside', 'nav'],
  replacement() {
    return '';
  },
});


/**
 * Suse has bad HTMl code snippets. We do our
 * best to parse them here.
 */
turndownService.addRule('remove', {
  filter: (node: any) => {
    if (node.nodeName !== 'PRE') return false;
    const firstChild = node.firstChild;
    if (firstChild.nodeName !== 'CODE') return false;

    const content = firstChild.textContent;
    if (content.startsWith('#') || content.startsWith('>')) return true;
    return true;
  },
  replacement: (content: string) => {
    content = content.replace('`#`', '#');
    content = content.replace('`>`', '>');
    content = content.replace('`sudo`', 'sudo');
    return `\n\`\`\`\n${content}\n\`\`\`\n`;
  },
});
/**
 * Add GFM support
 */
// turndownService.use(gfm);


export async function convertHTMLToMarkdown(html: string): Promise<string> {
  try {
    
    const select = cheerio.load(html);
    const title = select('title').text();
    const root = select('body');
    const md = turndownService.turndown(render(root));

    return md;

  } catch (error) {
    logger.error('Error converting HTML to Markdown:', error);
    return '';
  }
}