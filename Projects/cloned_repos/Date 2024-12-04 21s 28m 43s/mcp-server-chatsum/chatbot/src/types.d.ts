export interface ChatMessage {
  created_at: number;
  msg_id: string;
  room_id?: string;
  room_name?: string;
  room_avatar?: string;
  talker_id: string;
  talker_name?: string;
  talker_avatar?: string;
  content?: string;
  msg_type: number;
  url_title?: string;
  url_desc?: string;
  url_link?: string;
  url_thumb?: string;
}
