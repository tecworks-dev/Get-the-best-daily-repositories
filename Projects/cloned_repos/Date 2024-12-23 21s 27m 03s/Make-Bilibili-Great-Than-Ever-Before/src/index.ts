import { noop } from 'foxts/noop';
import { logger } from './logger';
import defuseSpyware from './modules/defuse-spyware';
import enhanceLive from './modules/enhance-live';
import fixCopyInCV from './modules/fix-copy-in-cv';
import noAd from './modules/no-ad';
import noP2P from './modules/no-p2p';
import noWebRTC from './modules/no-webtrc';
import optimizeHomepage from './modules/optimize-homepage';
import optimizeStory from './modules/optimize-story';
import playerVideoFit from './modules/player-video-fit';
import removeBlackBackdropFilter from './modules/remove-black-backdrop-filter';
import removeUselessUrlParams from './modules/remove-useless-url-params';
import useSystemFonts from './modules/use-system-fonts';
import type { OnXhrOpenHook } from './types';
import type { MakeBilibiliGreatThanEverBeforeHook, MakeBilibiliGreatThanEverBeforeModule, OnBeforeFetchHook } from './types';
import { onDOMContentLoaded } from './utils/on-load-event';

;((unsafeWindow) => {
  const modules: MakeBilibiliGreatThanEverBeforeModule[] = [
    defuseSpyware,
    enhanceLive,
    fixCopyInCV,
    noAd,
    noP2P,
    noWebRTC,
    optimizeHomepage,
    optimizeStory,
    playerVideoFit,
    removeBlackBackdropFilter,
    removeUselessUrlParams,
    useSystemFonts
  ];

  const styles: string[] = [];
  const onBeforeFetchHooks = new Set<OnBeforeFetchHook>();
  const onResponseHooks = new Set<(response: Response) => Response>();
  const onXhrOpenHooks = new Set<OnXhrOpenHook>();

  const hook: MakeBilibiliGreatThanEverBeforeHook = {
    addStyle(style: string) {
      styles.push(style);
    },
    onBeforeFetch(cb) {
      onBeforeFetchHooks.add(cb);
    },
    onResponse(cb) {
      onResponseHooks.add(cb);
    },
    onXhrOpen(cb) {
      onXhrOpenHooks.add(cb);
    }
  };

  const hostname = unsafeWindow.location.hostname;
  const pathname = unsafeWindow.location.pathname;

  for (const module of modules) {
    if (module.any) {
      logger.log(`[${module.name}] "any" ${unsafeWindow.location.href}`);
      module.any(hook);
    }
    switch (hostname) {
      case 'www.bilibili.com': {
        if (pathname.startsWith('/read/cv')) {
          if (module.onCV) {
            logger.log(`[${module.name}] "onCV" ${unsafeWindow.location.href}`);
            module.onCV(hook);
          }
        } else if (pathname.startsWith('/video/')) {
          if (module.onVideo) {
            logger.log(`[${module.name}] "onVideo" ${unsafeWindow.location.href}`);
            module.onVideo(hook);
          }
          if (module.onVideoOrBangumi) {
            logger.log(`[${module.name}] "onVideoOrBangumi" ${unsafeWindow.location.href}`);
            module.onVideoOrBangumi(hook);
          }
        } else if (pathname.startsWith('/bangumi/play/')) {
          if (module.onVideo) {
            logger.log(`[${module.name}] "onVideo" ${unsafeWindow.location.href}`);
            module.onVideo(hook);
          }
          if (module.onBangumi) {
            logger.log(`[${module.name}] "onBangumi" ${unsafeWindow.location.href}`);
            module.onBangumi(hook);
          }
          if (module.onVideoOrBangumi) {
            logger.log(`[${module.name}] "onVideoOrBangumi" ${unsafeWindow.location.href}`);
            module.onVideoOrBangumi(hook);
          }
        }
        break;
      }
      case 'live.bilibili.com': {
        if (module.onLive) {
          logger.log(`[${module.name}] "onLive" ${unsafeWindow.location.href}`);
          module.onLive(hook);
        }
        break;
      }
      case 't.bilibili.com': {
        if (module.onStory) {
          logger.log(`[${module.name}] "onStory" ${unsafeWindow.location.href}`);
          module.onStory(hook);
        }
        break;
      }
      // no default
    }
  }

  // Add Style
  onDOMContentLoaded(() => {
    const head = document.head || document.getElementsByTagName('head')[0];
    const style = document.createElement('style');
    style.setAttribute('type', 'text/css');
    style.textContent = styles.join('\n');
    head.appendChild(style);
  });

  // Override fetch
  (($fetch) => {
    unsafeWindow.fetch = function (...$fetchArgs) {
      let abortFetch = false;
      let fetchArgs: typeof $fetchArgs | null | Response = $fetchArgs;
      let mockResponse: Response | null = null;
      for (const obBeforeFetch of onBeforeFetchHooks) {
        try {
          fetchArgs = obBeforeFetch($fetchArgs);
          if (fetchArgs === null) {
            abortFetch = true;
            break;
          } else if (fetchArgs instanceof Response) {
            abortFetch = true;
            mockResponse = fetchArgs;
            break;
          }
        } catch (e) {
          logger.error('Failed to replace fetcherArgs', e, { fetchArgs: $fetchArgs });
        }
      }

      if (abortFetch) {
        logger.info('Fetch aborted', { fetchArgs, mockResponse });

        return Promise.resolve(mockResponse ?? new Response());
      }

      return Reflect.apply($fetch, this, $fetchArgs).then((response) => {
        for (const onResponse of onResponseHooks) {
          response = onResponse(response);
        }
        return response;
      });
    };
  })(unsafeWindow.fetch);

  (function (open) {
    unsafeWindow.XMLHttpRequest.prototype.open = function (this: XMLHttpRequest, ...$args: Parameters<XMLHttpRequest['open']>) {
      let xhrArgs: Parameters<XMLHttpRequest['open']> | null = $args;

      for (const onXhrOpen of onXhrOpenHooks) {
        try {
          xhrArgs = onXhrOpen($args, this);
          if (xhrArgs === null) {
            break;
          }
        } catch (e) {
          logger.error('Failed to replace P2P for XMLHttpRequest.prototype.open', e);
        }
      }

      if (xhrArgs === null) {
        logger.info('XHR aborted', { $args });
        this.send = noop;
        this.setRequestHeader = noop;
        return;
      }

      return Reflect.apply(open, this, xhrArgs);
    } as typeof XMLHttpRequest.prototype.open;
    // eslint-disable-next-line @typescript-eslint/unbound-method -- called with Reflect.apply
  }(unsafeWindow.XMLHttpRequest.prototype.open));
})(unsafeWindow);
