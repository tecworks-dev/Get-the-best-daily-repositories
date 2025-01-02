import type {main} from '../../wailsjs/go/models';

export type DeviceInfo = main.DeviceInfo;

export interface AppStatus {
    shizuku: boolean;
    nrfr: {
        installed: boolean;
        needUpdate: boolean;
    };
}

export type Step = 1 | 2 | 3 | 4 | 5;
