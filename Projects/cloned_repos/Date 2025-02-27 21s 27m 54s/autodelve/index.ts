import { download, readMarkdownFiles } from './lib/download';
import { ask } from './lib/ask';
import { connect } from './lib/discord';


// Get command line arguments
const args = process.argv.slice(2);
const command = args[0];
const input = args[1];

if (command === 'download' && input) {
  download(input);
} else if (command === 'ask' && input) {
  const answer = await ask(input);
  console.log(answer);
} else {
  console.log('Starting Discord bot...');
  connect();
}
