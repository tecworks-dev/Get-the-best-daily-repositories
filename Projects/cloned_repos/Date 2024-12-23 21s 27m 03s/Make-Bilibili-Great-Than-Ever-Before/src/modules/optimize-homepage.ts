import type { MakeBilibiliGreatThanEverBeforeModule } from '../types';

const optimizeHomepage: MakeBilibiliGreatThanEverBeforeModule = {
  name: 'optimize-homepage',
  description: '首页广告去除和样式优化',
  any({ addStyle }) {
    addStyle('.feed2 .feed-card:has(a[href*="cm.bilibili.com"]), .feed2 .feed-card:has(.bili-video-card:empty) { display: none } .feed2 .container > * { margin-top: 0 !important }');
  }
};

export default optimizeHomepage;
