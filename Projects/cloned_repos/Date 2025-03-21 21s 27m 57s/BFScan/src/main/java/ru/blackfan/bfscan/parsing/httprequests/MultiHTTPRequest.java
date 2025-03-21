package ru.blackfan.bfscan.parsing.httprequests;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import ru.blackfan.bfscan.parsing.httprequests.requestbody.ObjectRequestBody;
import ru.blackfan.bfscan.parsing.httprequests.requestbody.RequestBody;

public class MultiHTTPRequest {

    private List<HTTPRequest> requests;
    private String className;
    private String methodName;
    private static final String CONTENT_TYPE_HEADER = "content-type";
    public static final Map<String, String> CONTENT_TYPE_TO_ENC_TYPE = Map.of(
        "application/x-www-form-urlencoded", "form",
        "multipart/form-data", "multipart",
        "application/json", "json",
        "application/xml", "xml",
        "text/xml", "xml"
    );

    public MultiHTTPRequest(String apiHost, String apiBasePath, String className, String methodName) {
        this.className = className;
        this.methodName = methodName;
        this.requests = new ArrayList<>();
        HTTPRequest req = new HTTPRequest();
        req.basePath = apiBasePath;
        req.host = apiHost;
        this.requests.add(req);
    }

    public MultiHTTPRequest(MultiHTTPRequest multiRequest, String className, String methodName) {
        List<HTTPRequest> newRequests = new ArrayList<>();
        for (HTTPRequest request : multiRequest.requests) {
            newRequests.add(new HTTPRequest(request));
        }
        this.requests = newRequests;
        this.className = className;
        this.methodName = methodName;
    }

    public String getClassName() {
        return className;
    }

    public String getMethodName() {
        return methodName;
    }

    public void addPathParameter(String pathParameter) {
        for (HTTPRequest request : requests) {
            if (request.path.contains("{" + pathParameter + "}") && !request.pathParams.contains(pathParameter)) {
                request.pathParams.add(pathParameter);
            }
        }
    }

    public void addAdditionalInformation(String additionalInformation) {
        for (HTTPRequest request : requests) {
            request.additionalInformation.add(additionalInformation);
        }
    }

    public void putQueryParameter(String key, String value) {
        for (HTTPRequest request : requests) {
            request.queryParameters.put(key, value);
        }
    }

    public void putCookieParameter(String key, String value) {
        for (HTTPRequest request : requests) {
            request.cookieParameters.put(key, value);
        }
    }

    public void putHeader(String key, String value) {
        for (HTTPRequest request : requests) {
            if (key.toLowerCase().equals(CONTENT_TYPE_HEADER)) {
                String[] valueParts = value.split(";", 2);
                String[] contentTypes = valueParts[0].split(",");
                String selectedContentType = null;
                
                for (String contentType : contentTypes) {
                    contentType = contentType.trim().toLowerCase();
                    if (CONTENT_TYPE_TO_ENC_TYPE.containsKey(contentType)) {
                        selectedContentType = contentType;
                        break;
                    }
                }
                
                if (selectedContentType != null) {
                    setEncType(CONTENT_TYPE_TO_ENC_TYPE.get(selectedContentType));
                }
            }
            request.headers.put(key, value);
        }
    }

    public void putBodyParameter(String key, Object value) throws Exception {
        for (HTTPRequest request : requests) {
            if (request.requestBody == null) {
                request.requestBody = new ObjectRequestBody();
            }
            if (request.requestBody instanceof ObjectRequestBody) {
                ((ObjectRequestBody) request.requestBody).putBodyParameter(key, value);
            } else {
                request.requestBody = new ObjectRequestBody();
                ((ObjectRequestBody) request.requestBody).putBodyParameter(key, value);
            }
            if (request.method.equals("GET")) {
                request.method = "POST";
            }
        }
    }

    public void putBodyParameters(Map parameters) throws Exception {
        if (!parameters.isEmpty()) {
            for (HTTPRequest request : requests) {
                if (request.requestBody == null) {
                    request.requestBody = new ObjectRequestBody();
                }
                if (request.requestBody instanceof ObjectRequestBody) {
                    ((ObjectRequestBody) request.requestBody).putBodyParameters(parameters);
                } else {
                    request.requestBody = new ObjectRequestBody();
                    ((ObjectRequestBody) request.requestBody).putBodyParameters(parameters);
                }
                if (request.method.equals("GET")) {
                    request.method = "POST";
                }
            }
        }
    }

    public void setRequestBody(RequestBody requestBody) {
        for (HTTPRequest request : requests) {
            if (requestBody != null && request.method.equals("GET")) {
                request.method = "POST";
            }
            if (request.requestBody == null) {
                request.requestBody = requestBody;
            }
        }
    }

    public void setEncType(String encType) {
        for (HTTPRequest request : requests) {
            request.encType = encType;
        }
    }

    public void setHost(String host) {
        for (HTTPRequest request : requests) {
            request.host = host;
        }
    }

    public void setPath(String path, boolean query) {
        if (path.contains("?") && query) {
            String[] urlParts = path.split("\\?", 2);
            path = urlParts[0];
            String[] params = urlParts[1].split("&");
            for (String param : params) {
                String[] keyValue = param.split("=", 2);
                String qKey = keyValue[0];
                String qValue = keyValue.length > 1 ? keyValue[1] : "";
                if (!qKey.isEmpty() && !qValue.isEmpty()) {
                    putQueryParameter(qKey, qValue);
                }
            }
        }
        for (HTTPRequest request : requests) {
            request.path = path;
        }
    }

    public void setPaths(List<String> paths) {
        duplicateRequests(paths, "path");
    }

    public void setMethods(List<String> methods) {
        duplicateRequests(methods, "method");
    }

    public void setBasePaths(List<String> basePaths) {
        duplicateRequests(basePaths, "basePath");
    }

    public void setMethod(String method) {
        for (HTTPRequest request : requests) {
            request.method = method;
        }
    }

    private void duplicateRequests(List<String> values, String type) {
        if (requests.isEmpty() || values.isEmpty()) {
            return;
        }

        List<HTTPRequest> newRequests = new ArrayList<>();

        for (HTTPRequest request : new ArrayList<>(requests)) {
            for (String value : values) {
                HTTPRequest copy = new HTTPRequest(request);
                switch (type) {
                    case "path" ->
                        copy.path = value;
                    case "method" ->
                        copy.method = value;
                    case "basePath" ->
                        copy.basePath = value;
                }
                newRequests.add(copy);
            }
        }
        requests.clear();
        requests.addAll(newRequests);
    }

    public List<HTTPRequest> getRequests() {
        return this.requests;
    }
}
