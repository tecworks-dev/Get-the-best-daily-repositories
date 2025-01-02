import React, {useCallback, useEffect, useState} from 'react';
import {CheckApps, GetDevices, InstallNrfr, InstallShizuku, SelectDevice, StartShizuku} from '../wailsjs/go/main/App';
import type {AppStatus, DeviceInfo, Step} from './types';
import {ErrorBoundary} from './components/layout/ErrorBoundary';
import {TitleBar} from './components/layout/TitleBar';
import {StepIndicator} from './components/layout/StepIndicator';
import {ErrorMessage} from './components/layout/ErrorMessage';
import {DeviceSelection} from './components/steps/DeviceSelection';
import {AppCheck} from './components/steps/AppCheck';
import {AppInstall} from './components/steps/AppInstall';
import {ServiceStart} from './components/steps/ServiceStart';
import {Complete} from './components/steps/Complete';

function App() {
    const [step, setStep] = useState<Step>(1);
    const [devices, setDevices] = useState<DeviceInfo[]>([]);
    const [selectedDevice, setSelectedDevice] = useState<DeviceInfo | null>(null);
    const [appsStatus, setAppsStatus] = useState<AppStatus>({
        shizuku: false,
        nrfr: {
            installed: false,
            needUpdate: false
        }
    });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    const loadDevices = useCallback(async () => {
        try {
            setIsLoading(true);
            setError('');
            const deviceList = await GetDevices();
            if (!deviceList) {
                setDevices([]);
                return;
            }
            setDevices(Array.isArray(deviceList) ? deviceList : []);
        } catch (err: any) {
            setError(err.message || '获取设备列表失败');
            setDevices([]);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        loadDevices();
    }, [loadDevices]);

    const handleDeviceSelect = async (device: DeviceInfo) => {
        try {
            setIsLoading(true);
            setError('');
            await SelectDevice(device.serial);
            setSelectedDevice(device);
            const status = await CheckApps();
            setAppsStatus(status);
            setStep(2);
        } catch (err: any) {
            setError(err.message || '选择设备失败');
            setSelectedDevice(null);
        } finally {
            setIsLoading(false);
        }
    };

    const handleInstallApps = async () => {
        try {
            setIsLoading(true);
            setError('');

            // 只安装必需的应用
            if (!appsStatus.shizuku) {
                await InstallShizuku();
            }
            if (!appsStatus.nrfr.installed) {
                await InstallNrfr();
            }

            // 检查安装结果
            const newStatus = await CheckApps();
            setAppsStatus(newStatus);

            // 只验证必需应用是否安装成功
            if (!newStatus.shizuku || !newStatus.nrfr.installed) {
                setError('部分应用安装失败，请重试');
            }
        } catch (err: any) {
            setError(err.message || '安装应用失败');
        } finally {
            setIsLoading(false);
        }
    };

    const handleUpdateApp = async () => {
        try {
            setIsLoading(true);
            setError('');

            // 执行更新
            await InstallNrfr();

            // 检查更新结果
            const newStatus = await CheckApps();
            setAppsStatus(newStatus);
        } catch (err: any) {
            setError(err.message || '更新应用失败');
        } finally {
            setIsLoading(false);
        }
    };

    const handleStartService = async () => {
        try {
            setIsLoading(true);
            setError('');
            await StartShizuku();
            setStep(5);
        } catch (err: any) {
            setError(err.message || '启动服务失败');
        } finally {
            setIsLoading(false);
        }
    };

    const handleAppCheck = async () => {
        try {
            const status = await CheckApps();
            setAppsStatus(status);
        } catch (err: any) {
            setError(err.message || '检查应用状态失败');
        }
    };

    const handleNext = async (nextStep: Step) => {
        if (nextStep === 3) {
            // 在切换到步骤3之前先检查状态
            await handleAppCheck();
        }
        setStep(nextStep);
    };

    const renderStepContent = () => {
        if (!selectedDevice && step > 1) {
            return null;
        }

        switch (step) {
            case 1:
                return (
                    <DeviceSelection
                        devices={devices}
                        onSelect={handleDeviceSelect}
                        onRefresh={loadDevices}
                        isLoading={isLoading}
                    />
                );
            case 2:
                return (
                    <AppCheck
                        device={selectedDevice!}
                        appsStatus={appsStatus}
                        onNext={() => handleNext(3)}
                    />
                );
            case 3:
                return (
                    <AppInstall
                        device={selectedDevice!}
                        appsStatus={appsStatus}
                        isLoading={isLoading}
                        onInstall={handleInstallApps}
                        onUpdate={handleUpdateApp}
                        onNext={() => handleNext(4)}
                    />
                );
            case 4:
                return (
                    <ServiceStart
                        device={selectedDevice!}
                        isLoading={isLoading}
                        onStart={handleStartService}
                    />
                );
            case 5:
                return <Complete device={selectedDevice!}/>;
            default:
                return null;
        }
    };

    return (
        <ErrorBoundary>
            <div className="h-screen bg-transparent">
                <TitleBar/>
                <div className="h-[calc(100vh-2rem)] bg-gradient-to-b from-white/40 to-white/20 backdrop-blur-lg p-6">
                    <div className="max-w-md mx-auto">
                        <StepIndicator currentStep={step}/>
                        <ErrorMessage message={error} onClose={() => setError('')}/>
                        <div className="bg-white/30 backdrop-blur-md rounded-xl p-6">
                            {renderStepContent()}
                        </div>
                    </div>
                </div>
            </div>
        </ErrorBoundary>
    );
}

export default App;
