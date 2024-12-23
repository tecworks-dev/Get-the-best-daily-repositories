import { picker } from '@kit.CoreFileKit';

class PictureSelectUtil {

  async fileSelect(maxSelectNumber: number): Promise<Array<string>> {
    let photoSelectOptions = new picker.PhotoSelectOptions();
    photoSelectOptions.MIMEType = picker.PhotoViewMIMETypes.IMAGE_TYPE;
    photoSelectOptions.maxSelectNumber = maxSelectNumber;
    let photoPicker = new picker.PhotoViewPicker();
    try {
      let photoSelectResult = await photoPicker.select(photoSelectOptions);
      if (photoSelectResult && photoSelectResult.photoUris && photoSelectResult.photoUris.length > 0) {
        if (photoSelectResult.photoUris.length === 0) {
          return Array<string>();
        }
        return photoSelectResult.photoUris;
      } else {
        return Array<string>();
      }
    } catch (err) {
      console.error('SelectedImage failed', JSON.stringify(err))
      return Array<string>();
    }
  }
}

export default new PictureSelectUtil()