'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { Summary, Block } from '@/types';
import { Section } from './Section';
import { EditableTitle } from '../EditableTitle';

interface AISummaryProps {
  summary: Summary;
  onSummaryChange: (summary: Summary) => void;
}

export const AISummary: React.FC<AISummaryProps> = ({ summary, onSummaryChange }) => {
  const [selectedBlocks, setSelectedBlocks] = useState<string[]>([]);
  const [lastSelectedBlock, setLastSelectedBlock] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartBlock, setDragStartBlock] = useState<string | null>(null);
  const hiddenInputRef = useRef<HTMLTextAreaElement>(null);

  // History management
  const [history, setHistory] = useState<Summary[]>([summary]);
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState(0);
  const [isUndoRedoing, setIsUndoRedoing] = useState(false);

  // Add to history when summary changes
  useEffect(() => {
    if (!isUndoRedoing) {
      const newHistory = history.slice(0, currentHistoryIndex + 1);
      newHistory.push(summary);
      setHistory(newHistory);
      setCurrentHistoryIndex(newHistory.length - 1);
    }
    setIsUndoRedoing(false);
  }, [summary]);

  const handleUndo = useCallback(() => {
    if (currentHistoryIndex > 0) {
      setIsUndoRedoing(true);
      const newIndex = currentHistoryIndex - 1;
      setCurrentHistoryIndex(newIndex);
      onSummaryChange(history[newIndex]);
    }
  }, [currentHistoryIndex, history, onSummaryChange]);

  const handleRedo = useCallback(() => {
    if (currentHistoryIndex < history.length - 1) {
      setIsUndoRedoing(true);
      const newIndex = currentHistoryIndex + 1;
      setCurrentHistoryIndex(newIndex);
      onSummaryChange(history[newIndex]);
    }
  }, [currentHistoryIndex, history, onSummaryChange]);

  const getAllBlocks = () => {
    const allBlocks: { id: string; sectionKey: string }[] = [];
    Object.entries(summary).forEach(([sectionKey, section]) => {
      section.blocks.forEach(block => {
        allBlocks.push({ id: block.id, sectionKey });
      });
    });
    return allBlocks;
  };

  const findBlockAndSection = (blockId: string) => {
    for (const [sectionKey, section] of Object.entries(summary)) {
      const block = section.blocks.find(b => b.id === blockId);
      if (block) {
        return { block, sectionKey };
      }
    }
    return null;
  };

  const handleBlockNavigate = (blockId: string, direction: 'up' | 'down') => {
    const allBlocks = getAllBlocks();
    const currentIndex = allBlocks.findIndex(b => b.id === blockId);
    
    if (currentIndex === -1) return;
    
    let targetIndex: number;
    if (direction === 'up') {
      targetIndex = currentIndex > 0 ? currentIndex - 1 : currentIndex;
    } else {
      targetIndex = currentIndex < allBlocks.length - 1 ? currentIndex + 1 : currentIndex;
    }
    
    if (targetIndex !== currentIndex) {
      const targetBlock = allBlocks[targetIndex];
      setSelectedBlocks([targetBlock.id]);
      setLastSelectedBlock(targetBlock.id);
    }
  };

  const getBlockRange = (startId: string, endId: string) => {
    const allBlocks = getAllBlocks();
    const startIndex = allBlocks.findIndex(b => b.id === startId);
    const endIndex = allBlocks.findIndex(b => b.id === endId);
    
    if (startIndex === -1 || endIndex === -1) return [];
    
    const start = Math.min(startIndex, endIndex);
    const end = Math.max(startIndex, endIndex);
    
    return allBlocks.slice(start, end + 1).map(b => b.id);
  };

  const handleBlockMouseDown = (blockId: string, sectionKey: keyof Summary, e: React.MouseEvent<HTMLDivElement>) => {
    if (!e.shiftKey) {
      setDragStartBlock(blockId);
      setLastSelectedBlock(blockId);
      setSelectedBlocks([blockId]);
    }
    setIsDragging(true);
  };

  const handleBlockMouseEnter = (blockId: string, sectionKey: keyof Summary) => {
    if (isDragging && dragStartBlock) {
      const range = getBlockRange(dragStartBlock, blockId);
      setSelectedBlocks(range);
    }
  };

  const handleBlockMouseUp = (blockId: string, sectionKey: keyof Summary, e: React.MouseEvent<HTMLDivElement>) => {
    if (e.shiftKey && lastSelectedBlock) {
      const range = getBlockRange(lastSelectedBlock, blockId);
      setSelectedBlocks(range);
    }
    setIsDragging(false);
  };

  const handleBlockChange = (sectionKey: keyof Summary, blockId: string, newContent: string) => {
    onSummaryChange({
      ...summary,
      [sectionKey]: {
        ...summary[sectionKey],
        blocks: summary[sectionKey].blocks.map(block => 
          block.id === blockId ? { ...block, content: newContent } : block
        )
      }
    });
  };

  const handleBlockTypeChange = (blockId: string, newType: Block['type']) => {
    // Find the section key for this block
    let blockSectionKey: string | null = null;
    for (const [sectionKey, section] of Object.entries(summary)) {
      if (section.blocks.some(b => b.id === blockId)) {
        blockSectionKey = sectionKey;
        break;
      }
    }

    if (!blockSectionKey) return;

    onSummaryChange({
      ...summary,
      [blockSectionKey]: {
        ...summary[blockSectionKey],
        blocks: summary[blockSectionKey].blocks.map(block => 
          block.id === blockId ? { ...block, type: newType } : block
        )
      }
    });
  };

  const handleTitleChange = (sectionKey: keyof Summary, newTitle: string) => {
    console.log('Title change:', { sectionKey, newTitle });
    const updatedSummary = {
      ...summary,
      [sectionKey]: {
        ...summary[sectionKey],
        title: newTitle
      }
    };
    console.log('Updated summary:', updatedSummary);
    onSummaryChange(updatedSummary);
  };

  const handleKeyDown = (e: React.KeyboardEvent, blockId: string) => {
    // Find the section key for this block
    let blockSectionKey: string | null = null;
    for (const [sectionKey, section] of Object.entries(summary)) {
      if (section.blocks.some(b => b.id === blockId)) {
        blockSectionKey = sectionKey;
        break;
      }
    }

    if (!blockSectionKey) return;

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const currentBlock = summary[blockSectionKey].blocks.find(b => b.id === blockId);
      const currentBlockIndex = summary[blockSectionKey].blocks.findIndex(b => b.id === blockId);
      
      if (!currentBlock) return;
      
      const newId = Date.now().toString();
      const cursorPosition = (e.target as HTMLTextAreaElement).selectionStart;
      const textBeforeCursor = currentBlock.content.substring(0, cursorPosition);
      const textAfterCursor = currentBlock.content.substring(cursorPosition || 0);
      
      // Update the current block's content to only include text before cursor
      const updatedBlocks = [...summary[blockSectionKey].blocks];
      updatedBlocks[currentBlockIndex] = {
        ...currentBlock,
        content: textBeforeCursor
      };
      
      // Get the type of the current block for the new block
      const newBlockType = currentBlock.type === 'bullet' ? 'bullet' : 'text';
      
      // Insert new block after current block
      updatedBlocks.splice(currentBlockIndex + 1, 0, {
        id: newId,
        type: newBlockType,
        content: textAfterCursor,
        color: 'default'
      });
      
      onSummaryChange({
        ...summary,
        [blockSectionKey]: {
          ...summary[blockSectionKey],
          blocks: updatedBlocks
        }
      });
      
      setSelectedBlocks([newId]);
      setLastSelectedBlock(newId);
    } else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedBlocks.length > 1) {
      e.preventDefault();
      handleDeleteSelectedBlocks();
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      const cursorPosition = (e.target as HTMLTextAreaElement).selectionStart;
      const isAtStart = cursorPosition === 0;
      const isAtEnd = cursorPosition === (e.target as HTMLTextAreaElement).value.length;

      if ((e.key === 'ArrowUp' && isAtStart) || (e.key === 'ArrowDown' && isAtEnd)) {
        e.preventDefault();
        handleBlockNavigate(blockId, e.key === 'ArrowUp' ? 'up' : 'down');
      }
    }
  };

  const handleBlockDelete = (blockId: string) => {
    // Find the section key for this block
    let blockSectionKey: string | null = null;
    for (const [sectionKey, section] of Object.entries(summary)) {
      if (section.blocks.some(b => b.id === blockId)) {
        blockSectionKey = sectionKey;
        break;
      }
    }

    if (!blockSectionKey) return;

    const currentBlockIndex = summary[blockSectionKey].blocks.findIndex(b => b.id === blockId);
    const updatedBlocks = summary[blockSectionKey].blocks.filter(block => block.id !== blockId);
    
    onSummaryChange({
      ...summary,
      [blockSectionKey]: {
        ...summary[blockSectionKey],
        blocks: updatedBlocks
      }
    });

    // Select the previous block if it exists, otherwise the next block
    if (updatedBlocks.length > 0) {
      const newSelectedBlock = updatedBlocks[Math.max(0, currentBlockIndex - 1)];
      setSelectedBlocks([newSelectedBlock.id]);
      setLastSelectedBlock(newSelectedBlock.id);
    } else {
      setSelectedBlocks([]);
      setLastSelectedBlock(null);
    }
  };

  const getSelectedBlocksContent = useCallback(() => {
    return selectedBlocks
      .map(blockId => {
        for (const sectionKey of Object.keys(summary) as Array<keyof Summary>) {
          const block = summary[sectionKey].blocks.find(b => b.id === blockId);
          if (block) {
            return block.content;
          }
        }
        return '';
      })
      .filter(Boolean)
      .join('\n');
  }, [selectedBlocks, summary]);

  useEffect(() => {
    if (hiddenInputRef.current && selectedBlocks.length > 1) {
      const content = getSelectedBlocksContent();
      hiddenInputRef.current.value = content;
      hiddenInputRef.current.select();
    }
  }, [selectedBlocks, getSelectedBlocksContent]);

  useEffect(() => {
    const handleMouseUp = () => {
      setIsDragging(false);
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey)) {
        if (e.key === 'z') {
          e.preventDefault();
          if (e.shiftKey) {
            handleRedo();
          } else {
            handleUndo();
          }
        } else if (e.key === 'c') {
          const blockContents = selectedBlocks.map(blockId => {
            for (const sectionKey of Object.keys(summary) as Array<keyof Summary>) {
              const block = summary[sectionKey].blocks.find(b => b.id === blockId);
              if (block) {
                return block.content;
              }
            }
            return '';
          }).filter(Boolean);

          navigator.clipboard.writeText(blockContents.join('\n'));
        }
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedBlocks.length > 1) {
        e.preventDefault();
        handleDeleteSelectedBlocks();
      }
    };

    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedBlocks, summary, handleUndo, handleRedo]);

  const handleDeleteSelectedBlocks = () => {
    // Group selected blocks by section
    const blocksBySection = new Map<string, string[]>();
    selectedBlocks.forEach(blockId => {
      Object.entries(summary).forEach(([sectionKey, section]) => {
        if (section.blocks.some(b => b.id === blockId)) {
          const blocks = blocksBySection.get(sectionKey) || [];
          blocks.push(blockId);
          blocksBySection.set(sectionKey, blocks);
        }
      });
    });

    // Create new summary with blocks removed
    const newSummary = { ...summary };
    blocksBySection.forEach((blockIds, sectionKey) => {
      newSummary[sectionKey] = {
        ...newSummary[sectionKey],
        blocks: newSummary[sectionKey].blocks.filter(b => !blockIds.includes(b.id))
      };
    });

    onSummaryChange(newSummary);
    setSelectedBlocks([]);
    setLastSelectedBlock(null);
  };

  // Context menu state
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    visible: boolean;
  }>({ x: 0, y: 0, visible: false });

  // Close context menu when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      setContextMenu(prev => ({ ...prev, visible: false }));
    };
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    setContextMenu({
      x: e.clientX,
      y: e.clientY,
      visible: true
    });
  };

  const handleCopyBlocks = useCallback(() => {
    const content = getSelectedBlocksContent();
    navigator.clipboard.writeText(content);
    setContextMenu(prev => ({ ...prev, visible: false }));
  }, [getSelectedBlocksContent]);

  const handleDeleteBlocks = () => {
    handleDeleteSelectedBlocks();
    setContextMenu(prev => ({ ...prev, visible: false }));
  };

  const handleSectionDelete = (sectionKey: keyof Summary) => {
    const newSummary = { ...summary };
    delete newSummary[sectionKey];
    onSummaryChange(newSummary);
  };

  const handleAddSection = () => {
    const newSectionKey = `section${Object.keys(summary).length + 1}`;
    const newBlockId = Date.now().toString();
    const newSummary: Summary = {
      ...summary,
      [newSectionKey]: {
        title: 'New Section',
        blocks: [{
          id: newBlockId,
          type: 'text' as const,
          content: '',
          color: 'default' as const
        }]
      }
    };
    onSummaryChange(newSummary);
    
    // Select the new block
    setSelectedBlocks([newBlockId]);
    setLastSelectedBlock(newBlockId);
  };

  const convertToMarkdown = () => {
    let markdown = '';
    
    Object.entries(summary).forEach(([key, section]) => {
      if (key === 'title') {
        markdown = `# ${section.title || 'AI Enhanced Summary'}\n\n`;
      } else {
        markdown += `## ${section.title || key}\n\n`;
        section.blocks.forEach(block => {
          switch (block.type) {
            case 'heading1':
              markdown += `### ${block.content}\n\n`;
              break;
            case 'heading2':
              markdown += `#### ${block.content}\n\n`;
              break;
            case 'bullet':
              markdown += `- ${block.content}\n`;
              break;
            case 'text':
            default:
              markdown += `${block.content}\n\n`;
          }
        });
        // Add an extra newline after bullet lists
        if (section.blocks.some(block => block.type === 'bullet')) {
          markdown += '\n';
        }
      }
    });
    
    return markdown;
  };

  const handleExport = () => {
    const markdown = convertToMarkdown();
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${summary.title || 'ai-summary'}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="relative">
      <div className="flex justify-end mb-4 space-x-2">
        <button
          onClick={handleUndo}
          disabled={currentHistoryIndex === 0}
          className="p-2 hover:bg-gray-100 rounded disabled:opacity-50"
          title="Undo"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M3 7v6h6" />
            <path d="M21 17a9 9 0 00-9-9 9 9 0 00-6 2.3L3 13" />
          </svg>
        </button>
        <button
          onClick={handleRedo}
          disabled={currentHistoryIndex === history.length - 1}
          className="p-2 hover:bg-gray-100 rounded disabled:opacity-50"
          title="Redo"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 7v6h-6" />
            <path d="M3 17a9 9 0 019-9 9 9 0 016 2.3l3 2.7" />
          </svg>
        </button>
        <button
          onClick={handleAddSection}
          className="p-2 hover:bg-gray-100 rounded"
          title="Add new section"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 5v14" />
            <path d="M5 12h14" />
          </svg>
        </button>
      </div>
      
      {selectedBlocks.length > 1 && (
        <textarea
          ref={hiddenInputRef}
          className="sr-only"
          readOnly
          value={getSelectedBlocksContent()}
          tabIndex={-1}
        />
      )}
      
      {/* Context Menu */}
      {contextMenu.visible && selectedBlocks.length > 0 && (
        <div
          className="fixed z-50 bg-white shadow-lg rounded-lg py-1 min-w-[160px] border border-gray-200"
          style={{ 
            left: contextMenu.x, 
            top: contextMenu.y,
            transform: 'translate(-50%, -50%)'
          }}
          onClick={e => e.stopPropagation()}
        >
          <button
            className="w-full px-4 py-2 text-left hover:bg-gray-100 flex items-center space-x-2"
            onClick={handleCopyBlocks}
          >
            <span className="text-gray-600">📋</span>
            <span>Copy {selectedBlocks.length > 1 ? `${selectedBlocks.length} blocks` : 'block'}</span>
          </button>
          <button
            className="w-full px-4 py-2 text-left hover:bg-gray-100 text-red-600 flex items-center space-x-2"
            onClick={handleDeleteBlocks}
          >
            <span>🗑️</span>
            <span>Delete {selectedBlocks.length > 1 ? `${selectedBlocks.length} blocks` : 'block'}</span>
          </button>
        </div>
      )}

      <div className="flex items-center space-x-2 mb-6">
        <span className="text-2xl">✨</span>
        <h2 className="text-2xl font-semibold bg-gradient-to-r from-purple-600 to-blue-500 bg-clip-text text-transparent">
          AI Enhanced Summary
        </h2>
        <div className="ml-auto flex space-x-2">
          <button
            onClick={() => {
              const markdown = convertToMarkdown();
              navigator.clipboard.writeText(markdown);
            }}
            className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded-md flex items-center space-x-1"
          >
            <span>📋</span>
            <span>Copy as Markdown</span>
          </button>
          <button
            onClick={handleExport}
            className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded-md flex items-center space-x-1"
          >
            <span>📝</span>
            <span>Export as Markdown</span>
          </button>
        </div>
      </div>

      {Object.entries(summary).map(([key, section]) => (
        <Section
          key={key}
          section={section}
          sectionKey={key}
          selectedBlocks={selectedBlocks}
          onBlockTypeChange={handleBlockTypeChange}
          onBlockChange={(blockId, content) => handleBlockChange(key, blockId, content)}
          onBlockMouseDown={(blockId, e) => handleBlockMouseDown(blockId, key, e)}
          onBlockMouseEnter={(blockId) => handleBlockMouseEnter(blockId, key)}
          onBlockMouseUp={(blockId, e) => handleBlockMouseUp(blockId, key, e)}
          onKeyDown={handleKeyDown}
          onTitleChange={handleTitleChange}
          onSectionDelete={handleSectionDelete}
          onBlockDelete={handleBlockDelete}
          onContextMenu={handleContextMenu}
          onBlockNavigate={(blockId, direction) => handleBlockNavigate(blockId, direction)}
        />
      ))}
    </div>
  );
};
