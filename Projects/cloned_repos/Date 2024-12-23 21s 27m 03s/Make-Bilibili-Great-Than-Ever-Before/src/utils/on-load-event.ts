export function onDOMContentLoaded(callback: () => void) {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', callback, { once: true });
  } else {
    callback();
  }
}

export function onLoaded(callback: () => void) {
  if (document.readyState === 'complete') {
    callback();
  } else {
    // eslint-disable-next-line no-restricted-globals -- use sandboxed event handler
    window.addEventListener('load', callback, { once: true });
  }
}
