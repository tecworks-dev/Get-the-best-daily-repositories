import { Album, Asset, AssetType, ImageSize, SharedLink } from './types'
import dayjs from 'dayjs'
import { log } from './index'

class Immich {
  /**
   * Make a request to Immich API. We're not using the SDK to limit
   * the possible attack surface of this app.
   */
  async request (endpoint: string) {
    const res = await fetch(process.env.IMMICH_URL + '/api' + endpoint, {
      headers: {
        'x-api-key': process.env.API_KEY || ''
      }
    })
    if (res.status === 200) {
      const contentType = res.headers.get('Content-Type') || ''
      if (contentType.includes('application/json')) {
        return res.json()
      } else {
        return res
      }
    } else {
      log('Immich API status ' + res.status)
      console.log(await res.text())
    }
  }

  /**
   * Query Immich for the SharedLink metadata for a given key.
   * The key is what is returned in the URL when you create a share in Immich.
   */
  async getShareByKey (key: string) {
    const link = (await this.request('/shared-links/me?key=' + encodeURIComponent(key))) as SharedLink
    if (link) {
      if (link.expiresAt && dayjs(link.expiresAt) < dayjs()) {
        // This link has expired
        log('Expired link ' + key)
      } else {
        // Filter assets to exclude trashed assets
        link.assets = link.assets.filter(asset => !asset.isTrashed)
        link.assets.forEach(asset => { asset.key = key })
        return link
      }
    }
  }

  /**
   * Stream asset buffer data from Immich.
   *
   * For photos, you can request 'thumbnail' or 'original' size.
   * For videos, it is Immich's streaming quality, not the original quality.
   */
  async getAssetBuffer (asset: Asset, size?: ImageSize) {
    switch (asset.type) {
      case AssetType.image:
        size = size === ImageSize.thumbnail ? ImageSize.thumbnail : ImageSize.original
        return this.request('/assets/' + encodeURIComponent(asset.id) + '/' + size + '?key=' + encodeURIComponent(asset.key))
      case AssetType.video:
        return this.request('/assets/' + encodeURIComponent(asset.id) + '/video/playback?key=' + encodeURIComponent(asset.key))
    }
  }

  /**
   * Get the content-type of an Immich asset
   */
  async getContentType (asset: Asset) {
    const assetBuffer = await this.getAssetBuffer(asset)
    return assetBuffer.headers.get('Content-Type')
  }

  /**
   * Return the image data URL for a photo
   */
  photoUrl (key: string, id: string, size?: ImageSize) {
    return `/photo/${key}/${id}` + (size ? `?size=${size}` : '')
  }

  /**
   * Return the video data URL for a video
   */
  videoUrl (key: string, id: string) {
    return `/video/${key}/${id}`
  }

  /**
   * Check if a provided ID matches the Immich ID format
   */
  isId (id: string) {
    return !!id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/)
  }

  /**
   * Check if a provided key matches the Immich shared-link key format
   */
  isKey (key: string) {
    return !!key.match(/^[\w-]+$/)
  }
}

const immich = new Immich()

export default immich
