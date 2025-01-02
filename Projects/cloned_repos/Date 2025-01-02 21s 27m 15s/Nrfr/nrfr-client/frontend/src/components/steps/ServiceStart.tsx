import React, {useState} from 'react';
import type {DeviceInfo} from '../../types';

interface Props {
    device: DeviceInfo;
    isLoading: boolean;
    onStart: () => void;
}

export const ServiceStart: React.FC<Props> = ({device, isLoading, onStart}) => {
    const [isConfirmed, setIsConfirmed] = useState(false);

    const handleConfirm = () => {
        setIsConfirmed(true);
        onStart();
    };

    return (
        <div className="space-y-4">
            <h2 className="text-xl font-semibold text-center">启动服务</h2>
            <div className="text-center text-sm text-gray-600 mb-4">
                当前设备：{device.model || '未知设备'} ({device.serial})
            </div>
            <div className="p-4 bg-white/50 backdrop-blur-sm rounded-lg flex justify-between items-center">
                <span>准备启动 Shizuku 服务</span>
                {isLoading && (
                    <div className="animate-spin">
                        <svg className="w-4 h-4 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none"
                             viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor"
                                    strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor"
                                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                    </div>
                )}
            </div>
            <button
                className="w-full bg-blue-500/80 hover:bg-blue-600/80 text-white font-medium py-2 px-4 rounded-lg transition-all"
                onClick={handleConfirm}
                disabled={isLoading || isConfirmed}
            >
                {isLoading ? '启动中...' : isConfirmed ? '已确认' : '确认并启动'}
            </button>
        </div>
    );
};
