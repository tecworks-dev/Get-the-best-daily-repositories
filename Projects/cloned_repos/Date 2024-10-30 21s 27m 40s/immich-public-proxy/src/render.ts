import immich from './immich'
import { Response } from 'express-serve-static-core'
import { Asset, AssetType, ImageSize, SharedLink } from './types'

class Render {
  async assetBuffer (res: Response, asset: Asset, size?: ImageSize) {
    const data = await immich.getAssetBuffer(asset, size)
    if (data) {
      for (const header of ['content-type', 'content-length']) {
        res.set(header, data.headers[header])
      }
      res.send(Buffer.from(await data.arrayBuffer()))
    } else {
      res.status(404).send()
    }
  }

  /**
   * Render a gallery page for a given SharedLink, using EJS and lightGallery.
   *
   * @param res - ExpressJS Response
   * @param share - Immich `shared-link` containing the assets to show in the gallery
   * @param [openItem] - Immediately open a lightbox to the Nth item when the gallery loads
   */
  async gallery (res: Response, share: SharedLink, openItem?: number) {
    const items = []
    for (const asset of share.assets) {
      let video
      if (asset.type === AssetType.video) {
        // Populate the data-video property
        video = JSON.stringify({
          source: [
            {
              src: immich.videoUrl(share.key, asset.id),
              type: await immich.getContentType(asset)
            }
          ],
          attributes: {
            preload: false,
            controls: true
          }
        })
      }
      items.push({
        originalUrl: immich.photoUrl(share.key, asset.id),
        thumbnailUrl: immich.photoUrl(share.key, asset.id, ImageSize.thumbnail),
        video
      })
    }
    res.render('gallery', {
      items,
      openItem
    })
  }
}

const render = new Render()

export default render
