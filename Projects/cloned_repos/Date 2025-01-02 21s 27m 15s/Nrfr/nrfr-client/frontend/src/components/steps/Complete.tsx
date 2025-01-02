import React, {useCallback, useState} from 'react';
import type {DeviceInfo} from '../../types';
import {StartNrfr} from '../../../wailsjs/go/main/App';

interface Props {
    device: DeviceInfo;
}

export const Complete: React.FC<Props> = ({device}) => {
    const [isStarting, setIsStarting] = useState(false);
    const [error, setError] = useState('');

    const handleStartNrfr = useCallback(async () => {
        try {
            setIsStarting(true);
            setError('');
            await StartNrfr();
        } catch (err: any) {
            setError(err.message || '启动 Nrfr 失败');
        } finally {
            setIsStarting(false);
        }
    }, []);

    return (
        <div className="space-y-4">
            <h2 className="text-xl font-semibold text-center">设置完成</h2>
            <div className="text-center text-sm text-gray-600 mb-4">
                当前设备：{device.model || '未知设备'} ({device.serial})
            </div>
            <div className="p-4 bg-white/50 backdrop-blur-sm rounded-lg text-center">
                <div className="text-green-500 text-lg">✓ 全部完成</div>
                <div className="text-gray-600 mt-2">现在可以开始使用了</div>
            </div>

            {error && (
                <div className="p-4 bg-red-100 text-red-700 rounded-lg text-center">
                    {error}
                </div>
            )}

            <div className="space-y-2">
                <button
                    className="w-full bg-blue-500/80 hover:bg-blue-600/80 text-white font-medium py-2 px-4 rounded-lg transition-all flex items-center justify-center"
                    onClick={handleStartNrfr}
                    disabled={isStarting}
                >
                    {isStarting ? (
                        <>
                            <div className="animate-spin mr-2">
                                <svg className="w-4 h-4" xmlns="http://www.w3.org/2000/svg" fill="none"
                                     viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor"
                                            strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor"
                                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                            </div>
                            启动中...
                        </>
                    ) : (
                        '启动 Nrfr'
                    )}
                </button>
            </div>
        </div>
    );
};
