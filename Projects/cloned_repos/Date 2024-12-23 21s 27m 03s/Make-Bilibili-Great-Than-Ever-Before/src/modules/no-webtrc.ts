import { noop } from 'foxts/noop';
import { logger } from '../logger';
import type { MakeBilibiliGreatThanEverBeforeModule } from '../types';

// based on uBlock Origin's no-webrtc
// https://github.com/gorhill/uBlock/blob/6c228a8bfdcfc14140cdd3967270df28598c1aaf/src/js/resources/scriptlets.js#L2216
const noWebRTC: MakeBilibiliGreatThanEverBeforeModule = {
  name: 'no-webrtc',
  description: '通过禁用 WebRTC 防止叔叔省下棺材钱',
  any() {
    const rtcPcNames: string[] = [];

    if ('RTCPeerConnection' in unsafeWindow) {
      rtcPcNames.push('RTCPeerConnection');
    }
    if ('webkitRTCPeerConnection' in unsafeWindow) {
      rtcPcNames.push('webkitRTCPeerConnection');
    }
    if ('mozRTCPeerConnection' in unsafeWindow) {
      rtcPcNames.push('mozRTCPeerConnection');
    }

    const rtcDcNames: string[] = [];

    if ('RTCDataChannel' in unsafeWindow) {
      rtcDcNames.push('RTCDataChannel');
    }
    if ('webkitRTCDataChannel' in unsafeWindow) {
      rtcDcNames.push('webkitRTCDataChannel');
    }
    if ('mozRTCDataChannel' in unsafeWindow) {
      rtcDcNames.push('mozRTCDataChannel');
    }

    class MockDataChannel implements Pick<RTCDataChannel, 'close' | 'send' | 'addEventListener' | 'removeEventListener'> {
      close = noop;
      send = noop;
      addEventListener = noop;
      removeEventListener = noop;

      // eslint-disable-next-line @typescript-eslint/class-methods-use-this -- toString
      toString() {
        return '[object RTCDataChannel]';
      }
    }

    class MockPeerConnection implements Pick<RTCPeerConnection, 'close' | 'createDataChannel' | 'createOffer' | 'setRemoteDescription' | 'addEventListener' | 'removeEventListener'> {
      constructor(cfg: RTCConfiguration) {
        logger.log('Document tried to create an RTCPeerConnection', cfg);
      }

      close = noop;
      // eslint-disable-next-line @typescript-eslint/class-methods-use-this -- mock
      createDataChannel() {
        return new MockDataChannel() as RTCDataChannel;
      };

      createOffer = noop;
      setRemoteDescription = noop;
      addEventListener = noop;
      removeEventListener = noop;

      // eslint-disable-next-line @typescript-eslint/class-methods-use-this -- toString
      toString() {
        return '[object RTCPeerConnection]';
      }
    }

    for (const rtc of rtcPcNames) {
      Object.defineProperty(unsafeWindow, rtc, {
        get() {
          return MockPeerConnection;
        },
        set: noop,
        configurable: false,
        enumerable: true
      });
    }

    for (const dc of rtcDcNames) {
      Object.defineProperty(unsafeWindow, dc, {
        get() {
          return MockDataChannel;
        },
        set: noop,
        configurable: false,
        enumerable: true
      });
    }
  }
};

export default noWebRTC;
