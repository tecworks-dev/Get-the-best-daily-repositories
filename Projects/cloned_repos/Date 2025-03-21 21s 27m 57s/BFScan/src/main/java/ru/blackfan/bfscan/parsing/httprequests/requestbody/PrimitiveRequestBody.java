package ru.blackfan.bfscan.parsing.httprequests.requestbody;

public class PrimitiveRequestBody implements RequestBody {
    
    Object primitiveBody;
    
    public PrimitiveRequestBody(Object body) {
        primitiveBody = body;
    }
    
    public PrimitiveRequestBody(PrimitiveRequestBody body) {
        primitiveBody = body.getBody();
    }

    @Override
    public Type getType() {
        return Type.PRIMITIVE;
    }

    @Override
    public Object getBody() {
        return primitiveBody;
    }

    @Override
    public void setBody(Object body) {
        primitiveBody = body;
    }
    
}
