import { noop } from 'foxts/noop';
import type { MakeBilibiliGreatThanEverBeforeModule } from '../types';

declare global {
  interface Window {
    original?: {
      reprint: string
    }
  }
}

const fixCopyInCV: MakeBilibiliGreatThanEverBeforeModule = {
  name: 'fix-copy-in-cv',
  description: '修复文章复制功能',
  onCV() {
    if ('original' in unsafeWindow) {
      Object.defineProperty(unsafeWindow.original, 'reprint', {
        get() {
          return '1';
        },
        set: noop,
        configurable: false,
        enumerable: true
      });
    }

    const holder = document.querySelector('.article-holder');
    if (holder) {
      holder.classList.remove('unable-reprint');
      holder.addEventListener('copy', e => e.stopImmediatePropagation(), true);
    }
  }
};

export default fixCopyInCV;
