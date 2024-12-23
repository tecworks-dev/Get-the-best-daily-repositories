import type { MakeBilibiliGreatThanEverBeforeModule } from '../types';

const optimizeStory: MakeBilibiliGreatThanEverBeforeModule = {
  name: 'optimize-story',
  description: '动态页面优化',
  onStory({ addStyle }) {
    addStyle('html[wide] #app { display: flex; } html[wide] .bili-dyn-home--member { box-sizing: border-box;padding: 0 10px;width: 100%;flex: 1; } html[wide] .bili-dyn-content { width: initial; } html[wide] main { margin: 0 8px;flex: 1;overflow: hidden;width: initial; } #wide-mode-switch { margin-left: 0;margin-right: 20px; } .bili-dyn-list__item:has(.bili-dyn-card-goods), .bili-dyn-list__item:has(.bili-rich-text-module.goods) { display: none !important }');
    if (!localStorage.WIDE_OPT_OUT) {
      document.documentElement.setAttribute('wide', 'wide');
    }
    unsafeWindow.addEventListener('load', () => {
      const tabContainer = document.querySelector('.bili-dyn-list-tabs__list');
      const placeHolder = document.createElement('div');
      placeHolder.style.flex = '1';
      const switchButton = document.createElement('a');
      switchButton.id = 'wide-mode-switch';
      switchButton.className = 'bili-dyn-list-tabs__item';
      switchButton.textContent = '宽屏模式';
      switchButton.addEventListener('click', (e) => {
        e.preventDefault();
        if (localStorage.WIDE_OPT_OUT) {
          localStorage.removeItem('WIDE_OPT_OUT');
          document.documentElement.setAttribute('wide', 'wide');
        } else {
          localStorage.setItem('WIDE_OPT_OUT', '1');
          document.documentElement.removeAttribute('wide');
        }
      });
      tabContainer?.appendChild(placeHolder);
      tabContainer?.appendChild(switchButton);
    });
  }
};

export default optimizeStory;
