package ru.blackfan.bfscan.parsing.httprequests.requestbody;

public class RawRequestBody implements RequestBody {

    String rawBody;

    public RawRequestBody(String body) {
        rawBody = body;
    }

    public RawRequestBody(RequestBody body) {
        rawBody = (String) body.getBody();
    }

    @Override
    public Type getType() {
        return RequestBody.Type.RAW;
    }

    @Override
    public Object getBody() {
        return rawBody;
    }

    @Override
    public void setBody(Object body) {
        rawBody = (String) body;
    }
}
