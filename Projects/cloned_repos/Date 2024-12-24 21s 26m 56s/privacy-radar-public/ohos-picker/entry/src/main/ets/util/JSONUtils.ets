import { HashMap } from '@kit.ArkTS';

export class JSONUtils {
  static mapToJson(map: HashMap<string, string> | null): string {
    if (map == null) {
      return '';
    }
    let jsonObject: Record<string, string> = {};
    map.forEach((value, key) => {
      if (key !== undefined && value !== undefined) {
        jsonObject[key] = value;
      }
    });
    return JSON.stringify(jsonObject);
  }
}