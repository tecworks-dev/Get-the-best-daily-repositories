package ru.blackfan.bfscan.parsing.constants.file;

import java.io.InputStream;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import ru.blackfan.bfscan.helpers.KeyValuePair;

public class Properties {

    public static Set<KeyValuePair> process(String fileName, InputStream is) throws Exception {
        Set<KeyValuePair> keyValuePairs = new HashSet<>();
        java.util.Properties properties = new java.util.Properties();
        properties.load(is);
        for (Map.Entry<Object, Object> entry : properties.entrySet()) {
            keyValuePairs.add(new KeyValuePair((String) entry.getKey(), (String) entry.getValue()));
        }
        return keyValuePairs;
    }
}
