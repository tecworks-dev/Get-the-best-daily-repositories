// 去除鸿蒙字体，强制使用系统默认字体
import type { MakeBilibiliGreatThanEverBeforeModule } from '../types';

const useSystemFonts: MakeBilibiliGreatThanEverBeforeModule = {
  name: 'use-system-fonts',
  description: '去除鸿蒙字体，强制使用系统默认字体',
  any({ addStyle }) {
    document.querySelectorAll(String.raw`link[href*=\/font\/]`).forEach(x => x.remove());
    addStyle('html, body { font-family: initial !important; }');
  }
};

export default useSystemFonts;
