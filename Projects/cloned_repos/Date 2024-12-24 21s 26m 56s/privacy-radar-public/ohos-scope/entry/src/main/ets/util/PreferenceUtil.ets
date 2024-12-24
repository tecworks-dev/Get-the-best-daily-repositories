import dataPreferences from '@ohos.data.preferences';
import { Constants } from './Constants';
import Logger from './Logger';
import { ValueType } from '@kit.ArkData';


let context = getContext(this);
let preference: dataPreferences.Preferences;
let preferenceTemp: dataPreferences.Preferences;

class PreferenceUtil {
  /**
   * Read the specified Preferences persistence file and load the data into the Preferences instance.
   */
  async getPreferencesFromStorage() {
    try {
      preference = await dataPreferences.getPreferences(context, Constants.PREFERENCES_NAME);
    } catch (err) {
      Logger.error(`Failed to get preferences, Cause: ${err}`);
    }
  }

  /**
   * Deletes the specified Preferences persistence file from memory and removes the Preferences instance.
   */
  async deletePreferences() {
    try {
      await dataPreferences.deletePreferences(context, Constants.PREFERENCES_NAME);
    } catch(err) {
      Logger.error(`Failed to delete preferences, Cause: ${err}`);
    };
    preference = preferenceTemp;
  }

  async put(key: string,value: ValueType) {
    if (!preference) {
      await this.getPreferencesFromStorage();
    }
    // The fruit name and fruit quantity data entered by the user are saved to the cached Preference instance.
    try {
      await preference.put(key, value);
    } catch (err) {
      Logger.error(`Failed to put value, Cause: ${err}`);
    }
    // Store the Preference instance in the preference persistence file
    return await preference.flush();
  }

  /**
   * Get preference data.
   */
  async get(key: string) {
    if (!preference) {
      await this.getPreferencesFromStorage();
    }
    try {
      return await preference.get(key,'')
    } catch (err) {
      Logger.error(`Failed to get value, Cause: ${err}`);
    }
    return null;
  }

  async getString(key: string) {
    let value = '';
    if (!preference) {
      await this.getPreferencesFromStorage();
    }
    try {
      value = (await preference.get(key, '')).toString();
    } catch (err) {
      Logger.error(`Failed to get value, Cause: ${err}`);
    }
    return value;
  }

}

export default new PreferenceUtil()