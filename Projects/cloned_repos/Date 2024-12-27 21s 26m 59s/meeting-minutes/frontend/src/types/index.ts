export interface Message {
  id: string;
  content: string;
  timestamp: string;
}

export interface Transcript {
  id: string;
  text: string;
  timestamp: string;
}

export interface Block {
  id: string;
  type: 'text' | 'heading1' | 'heading2' | 'bullet';
  content: string;
  color?: 'default' | 'gray';
}

export interface Section {
  title: string;
  blocks: Block[];
}

export interface Summary {
  [key: string]: Section;
}
