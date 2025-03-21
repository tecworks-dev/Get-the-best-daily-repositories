package ru.blackfan.bfscan.config;

import java.io.InputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ConfigLoader {
    private static final Logger logger = LoggerFactory.getLogger(ConfigLoader.class);
    private static final String PROPERTIES_FILE = "application.properties";
    private static List<String> excludedPackages;
    private static String excludedSecretsRegexp;
    private static String excludedLinksRegexp;

    static {
        try (InputStream input = ConfigLoader.class.getClassLoader().getResourceAsStream(PROPERTIES_FILE)) {
            if (input == null) {
                logger.error("Configuration file not found: " + PROPERTIES_FILE);
            }
            Properties properties = new Properties();
            properties.load(input);
            String packages = properties.getProperty("excluded.packages", "");
            excludedPackages = Arrays.asList(packages.split(","));
            excludedSecretsRegexp = properties.getProperty("excluded.secretsRegexp", "");
            excludedLinksRegexp = properties.getProperty("excluded.linksRegexp", "");
        } catch (IOException e) {
            logger.error("Error loading configuration file", e);
        }
    }

    public static List<String> getExcludedPackages() {
        return excludedPackages;
    }

    public static String getExcludedSecretsRegexp() {
        return excludedSecretsRegexp;
    }

    public static String getExcludedLinksRegexp() {
        return excludedLinksRegexp;
    }
}