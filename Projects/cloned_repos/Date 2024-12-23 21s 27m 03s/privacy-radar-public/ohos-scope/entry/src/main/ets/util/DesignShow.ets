import display from '@ohos.display'

let context = getContext(this);

class DesignShow {

  PHONE_DESIGN_WIDTH: number = 750

  DESIGN_WIDTH: number = 360;
  DESIGN_HEIGHT: number = 780;

  vp(desginPx: number): number {
    let vp = desginPx
    try {
      vp = this.transDesignVp(desginPx,this.getPhoneSwVp(),this.PHONE_DESIGN_WIDTH)
    } catch (exception) {
      console.error('Failed to obtain the default display object. Code: ' + JSON.stringify(exception));
    }
    return vp
  }

  transDesignVp(pxValue: number, sw: number,designWidth: number): number {
    let dpValue = (pxValue / designWidth) * sw;
    return Number(dpValue.toFixed(2));
  }

  getPhoneSwVp (): number {
    let displayClass = display.getDefaultDisplaySync()
    return Math.min(px2vp(displayClass.width),px2vp(displayClass.height))
  }

  adaptDimension(value: number): number {
    let deviceDisplay = display.getDefaultDisplaySync()
    let widthScale = deviceDisplay.width / this.DESIGN_WIDTH;
    let virtualHeight = widthScale * this.DESIGN_HEIGHT;
    let designDim = Math.sqrt(this.DESIGN_WIDTH * this.DESIGN_WIDTH + this.DESIGN_HEIGHT * this.DESIGN_HEIGHT);
    let virtualDim = Math.sqrt(deviceDisplay.width * deviceDisplay.width + virtualHeight * virtualHeight);
    return virtualDim * value / designDim;
  }

  fp(desginPx: number): number  {
    return px2fp(this.adaptDimension(desginPx))
  }
  getFp(value: Resource): number  {
    let beforeVp = context.resourceManager.getNumber(value.id);
    return px2fp(this.adaptDimension(beforeVp))
  }

  getVp2(desginPx: number): number  {
    return px2vp(this.adaptDimension(desginPx))
  }
  getVp(value: Resource): number  {
    let beforeVp = context.resourceManager.getNumber(value.id);
    return px2vp(this.adaptDimension(beforeVp))
  }
  getPx(desginPx: number): number  {
    return this.adaptDimension(desginPx)
  }

}

export function lpx2fp(lpx: number): number {
  return px2fp(lpx2px(lpx))
}

export default new DesignShow()