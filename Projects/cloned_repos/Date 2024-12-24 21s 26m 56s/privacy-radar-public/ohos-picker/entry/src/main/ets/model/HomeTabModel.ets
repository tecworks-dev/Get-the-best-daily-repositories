export interface NavItem {
  icon: Resource;
  text: Resource;
  id: number;
}

export enum TabId {
  ALBUM,
  CONTACT,
  FILE,
  QR_SCAN,
  DEVICE_INFO,
  CALL_LOG,
  SMS,
  APP,
}

export const NavList: NavItem[] = [
  {
    icon: $r('app.media.ic_lock'),
    text: $r('app.string.album'),
    id: TabId.ALBUM
  },
  {
    icon: $r('app.media.ic_lock'),
    text: $r('app.string.contact'),
    id: TabId.CONTACT
  },
  {
    icon: $r('app.media.ic_lock'),
    text: $r('app.string.file'),
    id: TabId.FILE
  },
  {
    icon: $r('app.media.ic_lock'),
    text: $r('app.string.qr_scan'),
    id: TabId.QR_SCAN
  },
  {
    icon: $r('app.media.ic_lock'),
    text: $r('app.string.device_info'),
    id: TabId.DEVICE_INFO
  },
  {
    icon: $r('app.media.ic_lock'),
    text: $r('app.string.call_log'),
    id: TabId.CALL_LOG
  },
  {
    icon: $r('app.media.ic_lock'),
    text: $r('app.string.read_sms'),
    id: TabId.SMS
  },
  {
    icon: $r('app.media.ic_lock'),
    text: $r('app.string.app'),
    id: TabId.APP
  }
]
