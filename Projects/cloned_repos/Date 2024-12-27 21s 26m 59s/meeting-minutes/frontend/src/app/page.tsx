'use client';

import { useState, useEffect, useContext } from 'react';
import { Transcript, Summary } from '@/types';
import { EditableTitle } from '@/components/EditableTitle';
import { TranscriptView } from '@/components/TranscriptView';
import { RecordingControls } from '@/components/RecordingControls';
import { AISummary } from '@/components/AISummary';
import { useSidebar } from '@/components/Sidebar/SidebarProvider';
import MainNav from '@/components/MainNav';

export default function Home() {
  const [isRecording, setIsRecording] = useState(true);
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [showSummary, setShowSummary] = useState(false);
  const [barHeights, setBarHeights] = useState(['58%', '76%', '58%']);
  const [meetingTitle, setMeetingTitle] = useState('Intro call: AllFound');
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [aiSummary, setAiSummary] = useState<Summary>({
    overview: {
      title: 'AllFound Overview',
      blocks: [
        { id: '1', type: 'bullet', content: '100 employees, adding 20 more next quarter', color: 'default' },
        { id: '2', type: 'bullet', content: 'Office in San Francisco and Austin', color: 'gray' }
      ]
    },
    provider: {
      title: 'Current Provider (Tuesday.ai)',
      blocks: [
        { id: '3', type: 'bullet', content: 'Data input is too manual', color: 'default' },
        { id: '4', type: 'bullet', content: 'Too complex for non-technical team members', color: 'gray' },
        { id: '5', type: 'bullet', content: '$180 per employee per year ("too expensive")', color: 'default' }
      ]
    },
    requirements: {
      title: 'Their Requirements',
      blocks: [
        { id: '6', type: 'bullet', content: 'Finding a better employee engagement tool is "a priority for Q2"', color: 'default' },
        { id: '7', type: 'bullet', content: 'Need secure information sharing capabilities', color: 'gray' },
        { id: '8', type: 'bullet', content: 'One-way or two-way data sharing required, contingent on internal approval', color: 'gray' }
      ]
    },
    nextSteps: {
      title: 'Next steps',
      blocks: [
        { id: '9', type: 'bullet', content: 'Jess to send over information pack', color: 'gray' },
        { id: '10', type: 'bullet', content: 'Jess to send contract if we want to proceed', color: 'gray' },
        { id: '11', type: 'bullet', content: 'Catchup scheduled for next week to figure out contractual details', color: 'gray' }
      ]
    }
  });

  const [isCollapsed, setIsCollapsed] = useState(false);

  const { setCurrentMeeting } = useSidebar();

  useEffect(() => {
    setCurrentMeeting({ id: 'intro-call', title: meetingTitle });
  }, [meetingTitle, setCurrentMeeting]);

  useEffect(() => {
    if (isRecording) {
      const interval = setInterval(() => {
        setBarHeights(prev => {
          const newHeights = [...prev];
          newHeights[0] = Math.random() * 20 + 10 + 'px';
          newHeights[1] = Math.random() * 20 + 10 + 'px';
          newHeights[2] = Math.random() * 20 + 10 + 'px';
          return newHeights;
        });
      }, 300);

      return () => clearInterval(interval);
    }
  }, [isRecording]);

  useEffect(() => {
    if (isRecording) {
      const interval = setInterval(() => {
        const newTranscript = generateTranscript();
        setTranscripts(prev => [...prev, newTranscript]);
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [isRecording]);

  const handleRecordingStop = () => {
    setIsRecording(false);
    setShowSummary(true);
    generateAISummary();
  };

  const generateTranscript = () => {
    const texts = [
      "We need to focus on user engagement metrics.",
      "The current system is too complex for our needs.",
      "Security is a top priority for us.",
      "We're looking to implement this by Q2 next year.",
      "Integration with existing tools is crucial.",
    ];
    return {
      id: Date.now().toString(),
      text: texts[Math.floor(Math.random() * texts.length)],
      timestamp: new Date().toLocaleTimeString(),
    };
  };

  const generateAISummary = () => {
    // Keep existing summary state as is
  };

  const handleSummaryChange = (newSummary: Summary) => {
    console.log('Summary changed:', newSummary);
    setAiSummary(newSummary);
  };

  const handleTitleChange = (newTitle: string) => {
    setMeetingTitle(newTitle);
    setCurrentMeeting({ id: 'intro-call', title: newTitle });
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <MainNav />
      <div className="flex flex-1 overflow-hidden">
        {/* Left side - Transcript */}
        <div className="w-1/3 min-w-[300px] border-r border-gray-200 bg-white flex flex-col relative">
          <div className="p-4 border-b border-gray-200">
            <EditableTitle
              title={meetingTitle}
              isEditing={isEditingTitle}
              onEdit={setMeetingTitle}
              onEditingChange={setIsEditingTitle}
            />
          </div>
          <div className="flex-1 overflow-y-auto pb-32">
            <TranscriptView transcripts={transcripts} />
          </div>
          {/* Recording controls with improved positioning */}
          <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 z-10">
            <div className="bg-white rounded-full shadow-lg">
              <RecordingControls
                isRecording={isRecording}
                onRecordingStop={handleRecordingStop}
                onRecordingStart={() => setIsRecording(true)}
                barHeights={barHeights}
              />
            </div>
          </div>
        </div>

        {/* Right side - AI Summary */}
        <div className="flex-1 overflow-y-auto bg-white">
          {showSummary && (
            <div className="max-w-4xl mx-auto p-6">
              <AISummary summary={aiSummary} onSummaryChange={setAiSummary} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
