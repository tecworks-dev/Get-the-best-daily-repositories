package ru.blackfan.bfscan.parsing.constants.file;

import java.io.InputStream;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yaml.snakeyaml.constructor.SafeConstructor;
import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import ru.blackfan.bfscan.helpers.KeyValuePair;

public class Yml {
    private static final Logger logger = LoggerFactory.getLogger(Yml.class);
    private static final int MAX_DEPTH = 100;

    public static Set<KeyValuePair> process(String fileName, InputStream is) throws Exception {
        Set<KeyValuePair> keyValuePairs = new HashSet<>();
        
        LoaderOptions options = new LoaderOptions();
        options.setAllowDuplicateKeys(false);
        options.setMaxAliasesForCollections(100);
        options.setCodePointLimit(5 * 1024 * 1024);
        
        Yaml yaml = new Yaml(new SafeConstructor(options));
        
        try {
            Iterable<Object> yamlDocuments = yaml.loadAll(is);
            for (Object yamlData : yamlDocuments) {
                if (yamlData instanceof Map) {
                    parseYamlData("", (Map<Object, Object>) yamlData, keyValuePairs, 0);
                } else if (yamlData != null) {
                    logger.warn("Root YAML element in {} is not a map: {}", fileName, yamlData.getClass());
                }
            }
        } catch (Exception e) {
            logger.error("Error parsing YAML file: " + fileName, e);
            throw e;
        }
        
        return keyValuePairs;
    }

    private static void parseYamlData(String prefix, Map<Object, Object> data, Set<KeyValuePair> keyValuePairs, int depth) {
        if (depth > MAX_DEPTH) {
            logger.warn("Maximum YAML parsing depth ({}) exceeded", MAX_DEPTH);
            return;
        }
        
        for (Map.Entry<Object, Object> entry : data.entrySet()) {
            if (entry.getKey() == null) {
                continue;
            }
            
            String key = prefix.isEmpty() ? String.valueOf(entry.getKey()) : prefix + "." + entry.getKey();
            Object value = entry.getValue();
            
            if (value instanceof Map) {
                parseYamlData(key, (Map<Object, Object>) value, keyValuePairs, depth + 1);
            } else if (value instanceof Collection) {
                parseCollection(key, (Collection<?>) value, keyValuePairs, depth + 1);
            } else if (value != null) {
                keyValuePairs.add(new KeyValuePair(key, String.valueOf(value)));
            }
        }
    }
    
    private static void parseCollection(String prefix, Collection<?> collection, Set<KeyValuePair> keyValuePairs, int depth) {
        if (depth > MAX_DEPTH) {
            return;
        }
        
        int index = 0;
        for (Object item : collection) {
            String key = prefix + "[" + index + "]";
            if (item instanceof Map) {
                parseYamlData(key, (Map<Object, Object>) item, keyValuePairs, depth + 1);
            } else if (item instanceof Collection) {
                parseCollection(key, (Collection<?>) item, keyValuePairs, depth + 1);
            } else if (item != null) {
                keyValuePairs.add(new KeyValuePair(key, String.valueOf(item)));
            }
            index++;
        }
    }
}
