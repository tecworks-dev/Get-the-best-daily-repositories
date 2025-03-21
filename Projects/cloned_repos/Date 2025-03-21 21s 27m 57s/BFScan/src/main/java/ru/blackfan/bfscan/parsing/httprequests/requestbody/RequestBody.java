package ru.blackfan.bfscan.parsing.httprequests.requestbody;

public interface RequestBody {
    
    public enum Type { PRIMITIVE, OBJECT, RAW };
    
    public Type getType();
    public Object getBody();
    public void setBody(Object body);
    
}
