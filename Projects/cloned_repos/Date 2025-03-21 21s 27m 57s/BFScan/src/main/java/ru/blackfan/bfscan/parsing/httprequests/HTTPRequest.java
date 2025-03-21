package ru.blackfan.bfscan.parsing.httprequests;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import io.swagger.v3.oas.models.media.ArraySchema;
import io.swagger.v3.oas.models.media.BooleanSchema;
import io.swagger.v3.oas.models.media.Content;
import io.swagger.v3.oas.models.media.IntegerSchema;
import io.swagger.v3.oas.models.media.MediaType;
import io.swagger.v3.oas.models.media.Schema;
import io.swagger.v3.oas.models.media.StringSchema;
import io.swagger.v3.oas.models.media.XML;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.Operation;
import io.swagger.v3.oas.models.parameters.Parameter;
import io.swagger.v3.oas.models.parameters.RequestBody;
import io.swagger.v3.oas.models.PathItem;
import io.swagger.v3.oas.models.Paths;
import io.swagger.v3.oas.models.responses.ApiResponse;
import io.swagger.v3.oas.models.responses.ApiResponses;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.blackfan.bfscan.helpers.Helpers;
import ru.blackfan.bfscan.parsing.httprequests.requestbody.RequestBody.Type;

class HTTPRequest {

    private static final Logger logger = LoggerFactory.getLogger(HTTPRequest.class);

    static final String SPACE = " ";
    static final String CRLF = "\r\n";
    static final String BOUNDARY = "----WebKitFormBoundaryiohmaUfJHVC89E4H";
    public List<String> additionalInformation;
    public String method;
    public String basePath;
    public String path;
    public String httpVersion;
    public String host;
    public Map<String, Object> headers;
    public List<String> pathParams;
    public Map<String, Object> queryParameters;
    public ru.blackfan.bfscan.parsing.httprequests.requestbody.RequestBody requestBody;
    public Map<String, Object> cookieParameters;
    public String encType; // json,xml,multipart,form
    public final Gson gson = new GsonBuilder().setPrettyPrinting().create();

    public HTTPRequest(String argHost, String argBasePath) {
        method = "GET";
        path = "";
        basePath = argBasePath;
        httpVersion = "HTTP/1.1";
        host = argHost;
        headers = new HashMap();
        pathParams = new ArrayList();
        queryParameters = new HashMap();
        cookieParameters = new HashMap();
        encType = "get";
        additionalInformation = new ArrayList();
        requestBody = null;
    }

    public HTTPRequest() {
        this("localhost", "/");
    }

    public HTTPRequest(HTTPRequest copy) {
        method = copy.method;
        path = copy.path;
        basePath = copy.basePath;
        httpVersion = copy.httpVersion;
        host = copy.host;
        encType = copy.encType;
        additionalInformation = new ArrayList<>(copy.additionalInformation);
        headers = new HashMap<>(copy.headers);
        pathParams = new ArrayList();
        queryParameters = new HashMap();
        cookieParameters = new HashMap();
        requestBody = null;
    }

    public void addOpenApiPath(String className, String methodName, OpenAPI openApi) {
        if (openApi == null) {
            return;
        }

        String fullPath = fullPath();

        Operation operation = new Operation()
                .summary(methodName)
                .responses(new ApiResponses()
                        .addApiResponse("200", new ApiResponse().description("Success"))
                ).addTagsItem(className)
                .description(additionalInformation.stream()
                        .map(item -> "* " + item)
                        .collect(Collectors.joining("\n")));

        for (Map.Entry<String, Object> param : queryParameters.entrySet()) {
            operation.addParametersItem(new Parameter()
                    .name(param.getKey())
                    .in("query")
                    .required(false)
                    .schema(new StringSchema()._default(String.valueOf(param.getValue()))));
        }

        for (Map.Entry<String, Object> param : cookieParameters.entrySet()) {
            operation.addParametersItem(new Parameter()
                    .name(param.getKey())
                    .in("cookie")
                    .required(false)
                    .schema(new StringSchema()._default(String.valueOf(param.getValue()))));
        }

        for (Map.Entry<String, Object> header : headers.entrySet()) {
            operation.addParametersItem(new Parameter()
                    .name(header.getKey())
                    .in("header")
                    .required(true)
                    .schema(new StringSchema()._default(String.valueOf(header.getValue()))));
        }

        for (String pathParam : pathParams) {
            if (pathParam != null && path.contains("{" + pathParam + "}")) {
                operation.addParametersItem(
                        new Parameter()
                                .name(pathParam)
                                .in("path")
                                .required(true)
                                .schema(new StringSchema())
                );
            }
        }

        Pattern pattern = Pattern.compile("\\{(\\w+)}");
        Matcher matcher = pattern.matcher(path);
        while (matcher.find()) {
            String paramName = matcher.group(1);
            if (!pathParams.contains(paramName)) {
                operation.addParametersItem(new Parameter()
                        .name(paramName)
                        .in("path")
                        .required(true)
                        .schema(new StringSchema()));
            }

        }

        if (method.equalsIgnoreCase("POST") || method.equalsIgnoreCase("PUT") || method.equalsIgnoreCase("PATCH")) {
            Schema<?> bodySchema = new Schema<>().type("object");

            if (requestBody != null) {
                if (requestBody.getType() != null) {
                    switch (requestBody.getType()) {
                        case OBJECT -> {
                            bodySchema = new Schema<>().type("object");
                            if (encType.equals("xml")) {
                                bodySchema.xml(new XML().name("request"));
                            }
                            for (Map.Entry<String, Object> param : ((Map<String, Object>) requestBody.getBody()).entrySet()) {
                                Schema<?> schema;

                                if (param.getValue() == null) {
                                    schema = new StringSchema();
                                } else {
                                    switch (param.getValue().getClass().getSimpleName()) {
                                        case "Integer" ->
                                            schema = new IntegerSchema()._default((Integer) param.getValue());
                                        case "Boolean" ->
                                            schema = new BooleanSchema()._default((Boolean) param.getValue());
                                        case "List", "ArrayList" ->
                                            schema = processList((List<Object>) param.getValue());
                                        case "Map", "HashMap" -> {
                                            schema = processMap((Map<Object, Object>) param.getValue());
                                        }
                                        default ->
                                            schema = new StringSchema()._default(String.valueOf(param.getValue()));
                                    }
                                }

                                bodySchema.addProperty(param.getKey(), schema);
                            }
                        }
                        case PRIMITIVE -> {
                            if (requestBody.getBody() instanceof String) {
                                bodySchema = new Schema<>().type("string")._default(requestBody.getBody());
                            } else if (requestBody.getBody() instanceof Number) {
                                bodySchema = new Schema<>().type("number")._default(requestBody.getBody());
                            } else if (requestBody.getBody() instanceof Boolean) {
                                bodySchema = new Schema<>().type("boolean")._default(requestBody.getBody());
                            } else {
                                bodySchema = new Schema<>().type("object")._default(requestBody.getBody());
                            }
                        }
                        case RAW ->
                            bodySchema = new Schema<>().type("string")._default(requestBody.getBody());
                    }
                }
            }

            Content content = new Content();

            switch (encType) {
                case "xml" ->
                    content.addMediaType("application/xml", new MediaType().schema(bodySchema));
                case "form" ->
                    content.addMediaType("application/x-www-form-urlencoded", new MediaType().schema(bodySchema));
                case "multipart" ->
                    content.addMediaType("multipart/form-data", new MediaType().schema(bodySchema));
                default ->
                    content.addMediaType("application/json", new MediaType().schema(bodySchema));
            }

            RequestBody reqBody = new RequestBody()
                    .content(content)
                    .required(true);

            operation.requestBody(reqBody);
        }

        if (openApi.getPaths() == null) {
            openApi.setPaths(new Paths());
        }

        PathItem existingPathItem = openApi.getPaths().get(fullPath);
        if (existingPathItem == null) {
            existingPathItem = new PathItem();
        }
        if (!Helpers.isValidHttpMethod(method)) {
            if (existingPathItem.getPost() == null) {
                operation.setSummary(operation.getSummary() + " (custom http method " + method.toUpperCase() + ")");
                existingPathItem.operation(PathItem.HttpMethod.POST, operation);
            } else {
                logger.warn(className + "->" + methodName + ": Requests have the same path and method, OpenAPI specification will be overwritten");
            }
        } else {
            PathItem.HttpMethod httpMethod = PathItem.HttpMethod.valueOf(method.toUpperCase());
            if (!existingPathItem.readOperationsMap().containsKey(httpMethod)) {
                existingPathItem.operation(httpMethod, operation);
            } else {
                logger.warn(className + "->" + methodName + ": Requests have the same path and method, OpenAPI specification will be overwritten");
            }
        }
        openApi.path(fullPath, existingPathItem);
    }

    private Schema<Object> processMap(Map<Object, Object> nestedMap) {
        Schema<Object> objectSchema = new Schema<>().type("object");
        Map<String, Schema> properties = new HashMap<>();

        for (Map.Entry<Object, Object> entry : nestedMap.entrySet()) {
            Object value = entry.getValue();
            if (value instanceof Map<?, ?>) {
                properties.put(String.valueOf(entry.getKey()), processMap((Map<Object, Object>) value));
            } else if (value instanceof List<?>) {
                properties.put(String.valueOf(entry.getKey()), processList((List<Object>) value));
            } else if (value instanceof Integer) {
                properties.put(String.valueOf(entry.getKey()), new IntegerSchema());
            } else if (value instanceof Boolean) {
                properties.put(String.valueOf(entry.getKey()), new BooleanSchema());
            } else {
                properties.put(String.valueOf(entry.getKey()), new StringSchema());
            }
        }

        objectSchema.setProperties(properties);
        return objectSchema;
    }

    private Schema<?> processList(List<Object> list) {
        ArraySchema arraySchema = new ArraySchema();
        if (!list.isEmpty()) {
            Object firstElement = list.get(0);
            if (firstElement instanceof Map<?, ?>) {
                arraySchema.setItems(processMap((Map<Object, Object>) firstElement));
            } else if (firstElement instanceof List<?>) {
                arraySchema.setItems(processList((List<Object>) firstElement));
            } else if (firstElement instanceof Integer) {
                arraySchema.setItems(new IntegerSchema());
            } else if (firstElement instanceof Boolean) {
                arraySchema.setItems(new BooleanSchema());
            } else {
                arraySchema.setItems(new StringSchema());
            }
        } else {
            arraySchema.setItems(new StringSchema());
        }
        return arraySchema;
    }

    public String fullPath() {
        if (!basePath.startsWith("/")) {
            basePath = "/" + basePath;
        }
        if (!basePath.endsWith("/")) {
            basePath = basePath + "/";
        }
        return (path.startsWith("/") ? basePath.substring(0, basePath.length() - 1) : basePath) + path;
    }

    public String format() throws Exception {
        String out = "";

        out += method + SPACE + fullPath();
        out += path.contains("?") ? "&" : "?";
        for (Map.Entry<String, Object> entry : queryParameters.entrySet()) {
            out += entry.getKey() + "=" + entry.getValue() + "&";
        }
        out = out.substring(0, out.length() - 1);
        out += SPACE + httpVersion + CRLF;
        out += "Host: " + host + CRLF;
        if (!cookieParameters.isEmpty()) {
            out += "Cookie: ";
            for (Map.Entry<String, Object> entry : cookieParameters.entrySet()) {
                out += entry.getKey() + "=" + entry.getValue() + "; ";
            }
            out += CRLF;
        }
        out += "Connection: close" + CRLF;
        for (Map.Entry<String, Object> entry : headers.entrySet()) {
            out += entry.getKey() + ": " + entry.getValue() + CRLF;
        }
        if ((requestBody != null) && (requestBody.getBody() != null) && encType.equals("get")) {
            encType = "json";
        }
        switch (encType) {
            case "get" ->
                out += CRLF;
            case "form" -> {
                if (!out.contains("Content-Type")) {
                    out += "Content-Type: application/x-www-form-urlencoded" + CRLF;
                }
                out += CRLF;
                if (requestBody != null) {
                    if (requestBody.getType() == Type.OBJECT) {
                        for (Map.Entry<String, Object> entry : ((Map<String, Object>) requestBody.getBody()).entrySet()) {
                            if(entry.getValue() instanceof String) {
                                out += entry.getKey() + "=" + entry.getValue() + "&";
                            } else {
                                out += entry.getKey() + "=" + gson.toJson(entry.getValue()) + "&";
                            }
                        }
                        out = out.substring(0, out.length() - 1);
                    } else if (requestBody.getType() == Type.RAW) {
                        out += requestBody.getBody();
                    }
                }
            }
            case "multipart" -> {
                if (!out.contains("Content-Type")) {
                    out += "Content-Type: multipart/form-data; boundary=" + BOUNDARY + CRLF;
                }
                out += CRLF;
                if (requestBody != null) {
                    if (requestBody.getType() == Type.OBJECT) {
                        for (Map.Entry<String, Object> entry : ((Map<String, Object>) requestBody.getBody()).entrySet()) {
                            out += "--" + BOUNDARY + CRLF;
                            out += "Content-Disposition: form-data; name=\"" + entry.getKey() + "\"" + CRLF + CRLF;
                            out += entry.getKey() + CRLF;
                        }
                        out += "--" + BOUNDARY + "--";
                    }
                }
            }
            case "xml" -> {
                if (!out.contains("Content-Type")) {
                    out += "Content-Type: application/xml" + CRLF;
                }
                out += CRLF;
                if (requestBody != null) {
                    if (requestBody.getType() == Type.OBJECT) {
                        out += Helpers.convertToXML((Map<String, Object>) requestBody.getBody(), "request");
                    } else if (requestBody.getType() == Type.RAW) {
                        out += requestBody.getBody();
                    }
                } else {
                    if (!method.equals("GET")) {
                        out += "<request></request>";
                    }
                }
            }
            case "json" -> {
                if (!out.contains("Content-Type")) {
                    out += "Content-Type: application/json" + CRLF;
                }
                out += CRLF;
                if (requestBody != null) {
                    if (requestBody.getType() == Type.OBJECT) {
                        out += gson.toJson((HashMap<String, Object>) requestBody.getBody());
                    } else if (requestBody.getBody() != null && requestBody.getType() == Type.PRIMITIVE) {
                        out += gson.toJson(requestBody.getBody());
                    } else if (requestBody.getType() == Type.RAW) {
                        out += requestBody.getBody();
                    }
                } else {
                    if (!method.equals("GET")) {
                        out += "{}";
                    }
                }
            }
        }
        return out;
    }

}
