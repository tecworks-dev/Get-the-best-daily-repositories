export enum AssetType {
  image = 'IMAGE',
  video = 'VIDEO'
}

export interface Asset {
  id: string;
  key: string;
  type: AssetType;
  isTrashed: boolean;
}

export interface SharedLink {
  key: string;
  type: string;
  assets: Asset[];
  album?: {
    id: string;
  }
  expiresAt: string | null;
}

export interface Album {
  assets: Asset[]
}

export enum ImageSize {
  thumbnail = 'thumbnail',
  original = 'original'
}
