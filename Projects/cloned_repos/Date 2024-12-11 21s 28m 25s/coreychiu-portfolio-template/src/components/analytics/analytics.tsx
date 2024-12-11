import GoogleAnalytics from "./google-analytics";
import OpenPanelAnalytics from "./open-panel-analytics";
import { PlausibleAnalytics } from "./plausible-analytics";

export function Analytics() {
  if (process.env.NODE_ENV !== "production") {
    return null;
  }

  return (
    <>
      {/* openpanel analytics */}
      <OpenPanelAnalytics />

      {/* google analytics */}
      <GoogleAnalytics />

      {/* plausible analytics */}
      <PlausibleAnalytics />
    </>
  );
}
