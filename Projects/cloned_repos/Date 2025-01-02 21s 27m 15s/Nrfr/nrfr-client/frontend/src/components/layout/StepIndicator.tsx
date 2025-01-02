import React from 'react';

interface Props {
    currentStep: number;
}

export const StepIndicator: React.FC<Props> = ({currentStep}) => {
    return (
        <div className="mb-8">
            <div className="flex justify-between items-center">
                {[1, 2, 3, 4, 5].map((step) => (
                    <div
                        key={step}
                        className={`w-8 h-8 rounded-full flex items-center justify-center ${
                            step === currentStep
                                ? 'bg-blue-500 text-white'
                                : step < currentStep
                                    ? 'bg-green-500 text-white'
                                    : 'bg-gray-200 text-gray-600'
                        }`}
                    >
                        {step < currentStep ? '✓' : step}
                    </div>
                ))}
            </div>
        </div>
    );
};
