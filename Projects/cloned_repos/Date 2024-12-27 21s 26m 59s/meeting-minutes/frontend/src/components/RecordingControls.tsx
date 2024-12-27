'use client';

interface RecordingControlsProps {
  isRecording: boolean;
  barHeights: string[];
  onRecordingStop: () => void;
  onRecordingStart: () => void;
}

export const RecordingControls: React.FC<RecordingControlsProps> = ({
  isRecording,
  barHeights,
  onRecordingStop,
  onRecordingStart,
}) => {
  return (
    <div className="flex items-center space-x-3">
      {isRecording ? (
        <>
          {/* Recording animation bars */}
          <div className="flex items-center justify-center space-x-[3px] w-10 h-10">
            <div 
              className="w-[3px] bg-red-500 rounded-full transition-all duration-300 ease-in-out"
              style={{ height: barHeights[0] }}
            />
            <div 
              className="w-[3px] bg-red-500 rounded-full transition-all duration-300 ease-in-out"
              style={{ height: barHeights[1] }}
            />
            <div 
              className="w-[3px] bg-red-500 rounded-full transition-all duration-300 ease-in-out"
              style={{ height: barHeights[2] }}
            />
          </div>
          <button 
            onClick={onRecordingStop}
            className="w-12 h-12 bg-red-50 hover:bg-red-100 rounded-full flex items-center justify-center transition-colors duration-200 border-2 border-red-500"
          >
            <div className="w-4 h-4 bg-red-500 rounded-sm" />
          </button>
        </>
      ) : (
        <>
          {/* Paused indicator */}
          <div className="w-10 h-10 flex items-center justify-center">
            <div className="w-3 h-3 bg-gray-400 rounded-full animate-pulse" />
          </div>
          <button 
            onClick={onRecordingStart}
            className="w-12 h-12 bg-green-50 hover:bg-green-100 rounded-full flex items-center justify-center transition-colors duration-200 border-2 border-green-500"
          >
            <div className="w-0 h-0 border-t-[8px] border-t-transparent border-l-[12px] border-l-green-500 border-b-[8px] border-b-transparent ml-1" />
          </button>
        </>
      )}
    </div>
  );
};
