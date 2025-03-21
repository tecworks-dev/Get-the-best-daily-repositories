package ru.blackfan.bfscan.parsing.httprequests;

public class ParameterInfo {
    
    private String defaultValue;
    private String name;
    
    public ParameterInfo() {
        defaultValue = "";
        name = null;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDefaultValue() {
        return defaultValue;
    }

    public void setDefaultValue(String defaultValue) {
        this.defaultValue = defaultValue;
    }
    
}
