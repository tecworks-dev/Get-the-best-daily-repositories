import { noop } from 'foxts/noop';
import { logger } from '../logger';
import type { MakeBilibiliGreatThanEverBeforeModule } from '../types';

const rBackupCdn = /up[\w-]+\.bilivideo\.com/;

let prevLocationHref = '';
let prevCdnDomains: string[] = [];
function getCDNDomain() {
  if (prevLocationHref !== unsafeWindow.location.href || prevCdnDomains.length === 0) {
    try {
      const matched = rBackupCdn.exec(document.head.innerHTML);
      if (matched) {
        prevLocationHref = unsafeWindow.location.href;
        prevCdnDomains = matched;
      } else {
        logger.warn('Failed to get CDN domains from document.head.innerHTML, fallback to default CDN domain');
        prevLocationHref = unsafeWindow.location.href;
        prevCdnDomains = ['upos-sz-mirrorcoso1.bilivideo.com'];
        return 'upos-sz-mirrorcoso1.bilivideo.com';
      }
    } catch (e) {
      logger.error('Failed to get CDN domains from document.head.innerHTML, fallback to default CDN domain', e);
      prevLocationHref = unsafeWindow.location.href;
      prevCdnDomains = ['upos-sz-mirrorcoso1.bilivideo.com'];
      return 'upos-sz-mirrorcoso1.bilivideo.com';
    }
  }

  return prevCdnDomains[Math.floor(Math.random() * prevCdnDomains.length)];
}

const noP2P: MakeBilibiliGreatThanEverBeforeModule = {
  name: 'no-p2p',
  description: '防止叔叔用 P2P CDN 省下纸钱',
  any() {
    class MockPCDNLoader { }

    class MockBPP2PSDK {
      on = noop;
    }

    class MockSeederSDK { }

    Object.defineProperty(unsafeWindow, 'PCDNLoader', {
      get() {
        return MockPCDNLoader;
      },
      set: noop,
      enumerable: true,
      configurable: false
    });
    Object.defineProperty(unsafeWindow, 'BPP2PSDK', {
      get() {
        return MockBPP2PSDK;
      },
      set: noop,
      enumerable: true,
      configurable: false
    });
    Object.defineProperty(unsafeWindow, 'SeederSDK', {
      get() {
        return MockSeederSDK;
      },
      set: noop,
      enumerable: true,
      configurable: false
    });
  },
  onVideoOrBangumi({ onBeforeFetch, onXhrOpen }) {
    // Patch new Native Player
    (function (HTMLMediaElementPrototypeSrcDescriptor) {
      Object.defineProperty(unsafeWindow.HTMLMediaElement.prototype, 'src', {
        ...HTMLMediaElementPrototypeSrcDescriptor,
        set(value: string) {
          if (typeof value !== 'string') {
            value = String(value);
          }
          try {
            const result = replaceP2P(value, getCDNDomain(), 'HTMLMediaElement.prototype.src');
            value = typeof result === 'string' ? result : result.href;
          } catch (e) {
            logger.error('Failed to handle HTMLMediaElement.prototype.src setter', e, { value });
          }

          HTMLMediaElementPrototypeSrcDescriptor?.set?.call(this, value);
        }
      });
    }(Object.getOwnPropertyDescriptor(unsafeWindow.HTMLMediaElement.prototype, 'src')));

    onXhrOpen((xhrOpenArgs) => {
      try {
        xhrOpenArgs[1] = replaceP2P(
          xhrOpenArgs[1],
          getCDNDomain(),
          'XMLHttpRequest.prototype.open'
        );
      } catch (e) {
        logger.error('Failed to replace P2P for XMLHttpRequest.prototype.open', e, { xhrUrl: xhrOpenArgs[1] });
      }

      return xhrOpenArgs;
    });

    onBeforeFetch((fetchArgs: [RequestInfo | URL, RequestInit?]) => {
      let input = fetchArgs[0];
      if (typeof input === 'string' || input instanceof URL) {
        input = replaceP2P(input, getCDNDomain(), 'fetch');
      } else if (input instanceof Request) {
        input = new Request(replaceP2P(input.url, getCDNDomain(), 'fetch'), input);
      } else {
        const _: never = input;
        // input = replaceP2P(String(input), cdnDomain);
      }

      fetchArgs[0] = input;

      return fetchArgs;
    });
  }
};

export default noP2P;

function replaceP2P(url: string | URL, cdnDomain: string, meta = ''): string | URL {
  try {
    if (typeof url === 'string') {
      // early bailout for better performance
      if (!url.includes('.mcdn.bilivideo.cn') && !url.includes('.szbdyd.com')) {
        return url;
      }

      if (url.startsWith('//')) {
        url = unsafeWindow.location.protocol + url;
      }
      url = new URL(url, unsafeWindow.location.href);
    }
    const hostname = url.hostname;
    if (hostname.endsWith('.mcdn.bilivideo.cn')) {
      url.hostname = cdnDomain;
      url.port = '443';
      logger.info(`P2P replaced: ${hostname} -> ${cdnDomain}`, { meta });
    } else if (hostname.endsWith('.szbdyd.com')) {
      const xy_usource = url.searchParams.get('xy_usource');
      if (xy_usource) {
        url.hostname = xy_usource;
        url.port = '443';
        logger.info(`P2P replaced: ${hostname} -> ${xy_usource}`, { meta });
      }
    }

    return url;
  } catch (e) {
    logger.error(`Failed to replace P2P for URL (${url}):`, e);
    return url;
  }
}
