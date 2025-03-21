package ru.blackfan.bfscan.config;

public class ProcessorConfig {
    private static ProcessorConfig instance;
    private boolean minifiedAnnotationsSupport;

    private ProcessorConfig() {
        this.minifiedAnnotationsSupport = false;
    }

    public static ProcessorConfig getInstance() {
        if (instance == null) {
            instance = new ProcessorConfig();
        }
        return instance;
    }

    public boolean isMinifiedAnnotationsSupport() {
        return minifiedAnnotationsSupport;
    }

    public void setMinifiedAnnotationsSupport(boolean minifiedAnnotationsSupport) {
        this.minifiedAnnotationsSupport = minifiedAnnotationsSupport;
    }
} 