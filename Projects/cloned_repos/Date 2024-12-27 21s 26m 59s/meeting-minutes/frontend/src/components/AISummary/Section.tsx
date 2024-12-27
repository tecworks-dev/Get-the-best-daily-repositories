'use client';

import { Section as SectionType, Block } from '@/types';
import { BlockComponent } from './Block';
import { EditableTitle } from '../EditableTitle';
import { useState, useRef } from 'react';

interface SectionProps {
  section: SectionType;
  sectionKey: string;
  selectedBlocks: string[];
  onBlockTypeChange: (blockId: string, type: Block['type']) => void;
  onBlockChange: (blockId: string, content: string) => void;
  onBlockMouseDown: (blockId: string, e: React.MouseEvent<HTMLDivElement>) => void;
  onBlockMouseEnter: (blockId: string) => void;
  onBlockMouseUp: (blockId: string, e: React.MouseEvent<HTMLDivElement>) => void;
  onKeyDown: (e: React.KeyboardEvent, blockId: string) => void;
  onTitleChange?: (sectionKey: string, title: string) => void;
  onSectionDelete?: (sectionKey: string) => void;
  onBlockDelete: (blockId: string) => void;
  onContextMenu: (e: React.MouseEvent) => void;
  onBlockNavigate?: (blockId: string, direction: 'up' | 'down', cursorPosition: number) => void;
}

export const Section: React.FC<SectionProps> = ({
  section,
  sectionKey,
  selectedBlocks,
  onBlockTypeChange,
  onBlockChange,
  onBlockMouseDown,
  onBlockMouseEnter,
  onBlockMouseUp,
  onKeyDown,
  onTitleChange,
  onSectionDelete,
  onBlockDelete,
  onContextMenu,
  onBlockNavigate,
}) => {
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const titleInputRef = useRef<HTMLInputElement>(null);

  const handleTitleChange = (newTitle: string) => {
    if (onTitleChange) {
      onTitleChange(sectionKey, newTitle);
    }
  };

  const handleTitleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      setIsEditingTitle(false);
    }
  };

  return (
    <section>
      <div className="group relative mb-3">
        <EditableTitle
          title={section.title}
          isEditing={isEditingTitle}
          onStartEditing={() => setIsEditingTitle(true)}
          onFinishEditing={() => setIsEditingTitle(false)}
          onChange={handleTitleChange}
          onDelete={onSectionDelete ? () => onSectionDelete(sectionKey) : undefined}
        />
      </div>
      <div className="space-y-2">
        {section.blocks.map((block) => (
          <BlockComponent
            key={block.id}
            block={block}
            isSelected={selectedBlocks.includes(block.id)}
            onTypeChange={(type) => onBlockTypeChange(block.id, type)}
            onChange={(content) => onBlockChange(block.id, content)}
            onMouseDown={(e) => onBlockMouseDown(block.id, e)}
            onMouseEnter={() => onBlockMouseEnter(block.id)}
            onMouseUp={(e) => onBlockMouseUp(block.id, e)}
            onKeyDown={(e) => onKeyDown(e, block.id)}
            onDelete={() => onBlockDelete(block.id)}
            onContextMenu={onContextMenu}
            onNavigate={(direction, cursorPosition) => onBlockNavigate?.(block.id, direction, cursorPosition)}
          />
        ))}
      </div>
    </section>
  );
};
