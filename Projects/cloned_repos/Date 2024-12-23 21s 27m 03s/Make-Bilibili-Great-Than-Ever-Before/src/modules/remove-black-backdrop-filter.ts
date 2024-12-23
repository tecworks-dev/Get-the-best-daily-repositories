import type { MakeBilibiliGreatThanEverBeforeModule } from '../types';

const removeBlackBackdropFilter: MakeBilibiliGreatThanEverBeforeModule = {
  name: 'remove-black-backdrop-filter',
  description: '去除叔叔去世时的全站黑白效果',
  any({ addStyle }) {
    addStyle('html, body { -webkit-filter: none !important; filter: none !important; }');
  }
};

export default removeBlackBackdropFilter;
