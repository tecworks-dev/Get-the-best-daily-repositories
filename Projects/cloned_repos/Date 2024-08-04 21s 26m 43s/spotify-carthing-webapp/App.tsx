import { useEffect } from 'react';
import { observer } from 'mobx-react-lite';
import { GlobalStyles } from '@spotify-internal/encore-web';
import './App.scss';
import OtaUpdating from 'component/OtaUpdating/OtaUpdating';
import Main from 'component/Main';
import Setup from 'component/Setup/Setup';
import { useStore } from 'context/store';
import { SpotifySplash } from 'component/CarthingUIComponents';
import Settings from 'component/Settings/Settings';
import Onboarding from './component/Onboarding/Onboarding';

import { AppView } from 'store/ViewStore';
import RootErrorBoundary from 'component/RootErrorBoundary/RootErrorBoundary';

const App = () => {
  const { onboardingStore, viewStore, nightModeController } =
    useStore();
  useEffect(() => {
    onboardingStore.requestOnboardingStatus();
  }, [onboardingStore]);

  if (viewStore.appView === AppView.NOTHING) {
    return null;
  }

  const settingsEnabled = viewStore.appView !== AppView.LOGO;
  return (
    <div
      id="container"
      data-testid="app"
      style={{
        opacity: `${nightModeController.nightModeUiState.appOpacity}`,
      }}
    >
      <img id="corners" alt="" src="images/round-corners.svg" />
      <GlobalStyles />
      <RootErrorBoundary>
        {settingsEnabled && <Settings />}
        {
          {
            [AppView.LOGO]: <SpotifySplash />,
            [AppView.SETUP]: <Setup />,
            [AppView.OTA]: <OtaUpdating />,
            [AppView.ONBOARDING]: <Onboarding />,
            [AppView.MAIN]: <Main />,
          }[viewStore.appView]
        }
      </RootErrorBoundary>
    </div>
  );
};
export default observer(App);
