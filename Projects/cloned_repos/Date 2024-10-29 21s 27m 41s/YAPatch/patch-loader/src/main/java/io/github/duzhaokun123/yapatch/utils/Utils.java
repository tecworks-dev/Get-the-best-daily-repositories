package io.github.duzhaokun123.yapatch.utils;

import org.json.JSONArray;
import org.json.JSONException;

public class Utils {
    public static String[] fromJsonArray(JSONArray jsonArray) throws JSONException {
        String[] result = new String[jsonArray.length()];
        for (int i = 0; i < jsonArray.length(); i++) {
            result[i] = jsonArray.getString(i);
        }
        return result;
    }
}
