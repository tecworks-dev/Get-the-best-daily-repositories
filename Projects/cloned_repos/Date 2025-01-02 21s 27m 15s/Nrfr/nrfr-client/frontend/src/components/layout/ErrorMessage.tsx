import React from 'react';

interface Props {
    message: string;
    onClose: () => void;
}

export const ErrorMessage: React.FC<Props> = ({message, onClose}) => {
    if (!message) return null;

    return (
        <div className="mb-4 p-4 bg-red-100 text-red-700 rounded-lg">
            {message}
            <button
                className="ml-2 text-red-500 hover:text-red-600"
                onClick={onClose}
            >
                关闭
            </button>
        </div>
    );
};
