import React, {useState} from 'react';
import type {AppStatus, DeviceInfo} from '../../types';

interface Props {
    device: DeviceInfo;
    appsStatus: AppStatus;
    onNext: () => void;
}

export const AppCheck: React.FC<Props> = ({device, appsStatus, onNext}) => {
    const [isConfirmed, setIsConfirmed] = useState(false);

    const handleConfirm = () => {
        setIsConfirmed(true);
        onNext();
    };

    return (
        <div className="space-y-4">
            <h2 className="text-xl font-semibold text-center">检查应用</h2>
            <div className="text-center text-sm text-gray-600 mb-4">
                当前设备：{device.model || '未知设备'} ({device.serial})
            </div>
            <div className="space-y-2">
                <div className="p-4 bg-white/50 backdrop-blur-sm rounded-lg">
                    <div className="flex justify-between items-center">
                        <span>Shizuku</span>
                        <div className="flex items-center">
                            <span className={appsStatus.shizuku ? 'text-green-500' : 'text-red-500'}>
                                {appsStatus.shizuku ? '已安装' : '未安装'}
                            </span>
                            {appsStatus.shizuku && (
                                <svg className="w-5 h-5 ml-2 text-green-500" fill="none" stroke="currentColor"
                                     viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                          d="M5 13l4 4L19 7"/>
                                </svg>
                            )}
                        </div>
                    </div>
                </div>
                <div className="p-4 bg-white/50 backdrop-blur-sm rounded-lg">
                    <div className="flex justify-between items-center">
                        <span>Nrfr</span>
                        <div className="flex items-center">
                            <span className={appsStatus.nrfr.installed ? 'text-green-500' : 'text-red-500'}>
                                {appsStatus.nrfr.installed ? '已安装' : '未安装'}
                            </span>
                            {appsStatus.nrfr.installed && (
                                <svg className="w-5 h-5 ml-2 text-green-500" fill="none" stroke="currentColor"
                                     viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                          d="M5 13l4 4L19 7"/>
                                </svg>
                            )}
                        </div>
                    </div>
                </div>
            </div>
            <button
                className="w-full bg-blue-500/80 hover:bg-blue-600/80 text-white font-medium py-2 px-4 rounded-lg transition-all"
                onClick={handleConfirm}
                disabled={isConfirmed}
            >
                {isConfirmed ? '已确认' : '确认并继续'}
            </button>
        </div>
    );
};

