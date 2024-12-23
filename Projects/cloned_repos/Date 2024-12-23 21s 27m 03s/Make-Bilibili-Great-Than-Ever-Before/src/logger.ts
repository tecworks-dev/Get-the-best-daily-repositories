/* eslint-disable no-restricted-globals -- logger */
/* eslint-disable no-console -- logger */
export const logger = {
  log(...args: unknown[]) {
    console.log('[make-bilibili-great-than-ever-before]', ...args);
  },
  error(...args: unknown[]) {
    console.error('[make-bilibili-great-than-ever-before]', ...args);
  },
  warn(...args: unknown[]) {
    console.warn('[make-bilibili-great-than-ever-before]', ...args);
  },
  info(...args: unknown[]) {
    console.info('[make-bilibili-great-than-ever-before]', ...args);
  },
  debug(...args: unknown[]) {
    console.debug('[make-bilibili-great-than-ever-before]', ...args);
  },
  trace(...args: unknown[]) {
    console.trace('[make-bilibili-great-than-ever-before]', ...args);
  }
};
