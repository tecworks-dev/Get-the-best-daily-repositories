package ru.blackfan.bfscan.parsing.httprequests.requestbody;

import java.util.HashMap;
import java.util.Map;

public class ObjectRequestBody implements RequestBody {
    public Map<String, Object> bodyParameters;
    
    public ObjectRequestBody(Map<String, Object> body) {
        bodyParameters = body;
    }
    
    public ObjectRequestBody(ObjectRequestBody body) {
        bodyParameters = new HashMap<>(body.bodyParameters);
    }
    
    public ObjectRequestBody() {
        bodyParameters = new HashMap<>();
    }
    
    public void putBodyParameter(String key, Object value) {
        bodyParameters.put(key, value);
    }
    
    public void putBodyParameters(Map parameters) {
        bodyParameters.putAll(parameters);
    }

    @Override
    public Type getType() {
        return Type.OBJECT;
    }

    @Override
    public Object getBody() {
        return bodyParameters;
    }

    @Override
    public void setBody(Object body) {
        bodyParameters = (HashMap<String, Object>) body;
    }
}
