import type { MakeBilibiliGreatThanEverBeforeModule } from '../types';
import { getUrlFromRequest } from '../utils/get-url-from-request';

declare global {
  interface Window {
    disableMcdn?: boolean
  }
}

// const mcdnRegexp = /[\dxy]+\.mcdn\.bilivideo\.cn:\d+/;
const qualityRegexp = /(live-bvc\/\d+\/live_\d+_\d+)_\w+/;

const enhanceLive: MakeBilibiliGreatThanEverBeforeModule = {
  name: 'enhance-live',
  description: '增强直播（原画画质、其他修复）',
  onLive({ addStyle, onBeforeFetch, onResponse }) {
    let forceHighestQuality = true;
    let recentErrors = 0;

    setInterval(() => (recentErrors > 0 ? recentErrors / 2 : null), 10000);

    // 还得帮叔叔修 bug，唉
    addStyle('div[data-cy=EvaRenderer_LayerWrapper]:has(.player) { z-index: 999999; }');

    // 去台标
    addStyle('.web-player-icon-roomStatus { display: none !important; }');

    // 修复直播画质
    onBeforeFetch((fetchArgs) => {
      const url = getUrlFromRequest(fetchArgs[0]);
      if (url == null) {
        return fetchArgs;
      }
      try {
        // if (mcdnRegexp.test(url) && disableMcdn) {
        //   return Promise.reject();
        // }
        if (qualityRegexp.test(url) && forceHighestQuality) {
          fetchArgs[0] = url
            .replace(qualityRegexp, '$1')
            .replaceAll(/(\d+)_(?:mini|pro)hevc/g, '$1');
        }

        return fetchArgs;
      } catch {
        return fetchArgs;
      }
    });

    onResponse((response) => {
      if (response.url.endsWith('.m3u8') || response.url.endsWith('.m4s')) {
        if (!response.ok) recentErrors++;
        if (recentErrors >= 5 && forceHighestQuality) {
          recentErrors = 0;
          forceHighestQuality = false;
          GM.notification(
            '[Make Bilibili Great Then Ever Before] 已为您自动切换至播放器上选择的清晰度.',
            '最高清晰度可能不可用'
          );
        }
      }
      return response;
    });
  }
};

export default enhanceLive;
