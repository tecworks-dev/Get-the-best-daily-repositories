export interface MakeBilibiliGreatThanEverBeforeModule {
  name: string,
  description: string,
  any?: (hook: MakeBilibiliGreatThanEverBeforeHook) => void,
  onVideo?: (hook: MakeBilibiliGreatThanEverBeforeHook) => void,
  onLive?: (hook: MakeBilibiliGreatThanEverBeforeHook) => void,
  onCV?: (hook: MakeBilibiliGreatThanEverBeforeHook) => void,
  onStory?: (hook: MakeBilibiliGreatThanEverBeforeHook) => void,
  onBangumi?: (hook: MakeBilibiliGreatThanEverBeforeHook) => void,
  onVideoOrBangumi?: (hook: MakeBilibiliGreatThanEverBeforeHook) => void
}

export type OnBeforeFetchHook = (fetchArgs: [requestInfo: RequestInfo | URL, requestInit?: RequestInit]) => [requestInfo: RequestInfo | URL, requestInit?: RequestInit] | null | Response;
export type OnXhrOpenHook = (xhrOpenArgs: Parameters<XMLHttpRequest['open']>, xhr: XMLHttpRequest) => Parameters<XMLHttpRequest['open']> | null;

export interface MakeBilibiliGreatThanEverBeforeHook {
  addStyle(this: void, css: string): void,
  onBeforeFetch(this: void, cb: OnBeforeFetchHook): void,
  onResponse(this: void, cb: (response: Response) => Response): void,
  onXhrOpen(this: void, cb: OnXhrOpenHook): void
};
