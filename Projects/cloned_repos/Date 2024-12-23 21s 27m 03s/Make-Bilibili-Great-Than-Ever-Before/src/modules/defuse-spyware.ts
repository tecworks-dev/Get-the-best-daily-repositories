import { noop, trueFn } from 'foxts/noop';
import { getUrlFromRequest } from '../utils/get-url-from-request';
import { createMockClass } from '../utils/mock-class';
import type { MakeBilibiliGreatThanEverBeforeModule } from '../types';

const defuseSpyware: MakeBilibiliGreatThanEverBeforeModule = {
  name: 'defuse-spyware',
  description: '禁用叔叔日志上报和用户跟踪的无限请求风暴',
  any({ onBeforeFetch, onXhrOpen }) {
    Object.defineProperty(unsafeWindow.navigator, 'sendBeacon', {
      get() {
        return trueFn;
      },
      set: noop,
      configurable: false,
      enumerable: true
    });

    const SentryHub = class { bindClient = noop; };

    const fakeSentry = {
      SDK_NAME: 'sentry.javascript.browser',
      SDK_VERSION: '0.0.1145141919810',
      BrowserClient: class {},
      Hub: SentryHub,
      Integrations: {
        Vue: class {},
        GlobalHandlers: class {},
        InboundFilters: class {}
      },
      init: noop,
      configureScope: noop,
      getCurrentHub: () => new SentryHub(),
      setContext: noop,
      setExtra: noop,
      setExtras: noop,
      setTag: noop,
      setTags: noop,
      setUser: noop,
      wrap: noop
    };

    Object.defineProperty(unsafeWindow, 'Sentry', {
      get() {
        return fakeSentry;
      },
      set: noop,
      configurable: false,
      enumerable: true
    });

    const mReport = createMockClass('MReporter');
    Object.defineProperty(unsafeWindow, 'MReporter', {
      get() {
        return mReport;
      },
      set: noop,
      configurable: false,
      enumerable: true
    });

    const reporterPb = class {
      click = noop;
      custom = noop;
      exposure = noop;
      report = noop;
      tech = noop;
      pv = noop;
    };
    Object.defineProperty(unsafeWindow, 'ReporterPb', {
      get() {
        return reporterPb;
      },
      set: noop,
      configurable: false,
      enumerable: true
    });

    const biliUserFp = {
      init: noop,
      queryUserLog() {
        return [];
      }
    };

    Object.defineProperty(unsafeWindow, '__biliUserFp__', {
      get() { return biliUserFp; },
      set: noop,
      configurable: false,
      enumerable: true
    });
    Object.defineProperty(unsafeWindow, '__USER_FP_CONFIG__', { get: noop, set: noop, configurable: false, enumerable: true });
    Object.defineProperty(unsafeWindow, '__MIRROR_CONFIG__', { get: noop, set: noop, configurable: false, enumerable: true });

    onBeforeFetch((fetchArgs) => {
      const url = getUrlFromRequest(fetchArgs[0]);

      if (typeof url === 'string' && url.includes('data.bilibili.com')) {
        return new Response();
      };

      return fetchArgs;
    });

    onXhrOpen((args) => {
      let url = args[1];
      if (typeof url !== 'string') {
        url = url.href;
      }

      if (url.includes('data.bilibili.com')) {
        return null;
      }

      return args;
    });
  }
};

export default defuseSpyware;
