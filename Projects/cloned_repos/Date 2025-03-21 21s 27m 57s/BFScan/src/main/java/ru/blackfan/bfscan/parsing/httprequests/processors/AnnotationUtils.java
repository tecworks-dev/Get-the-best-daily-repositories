package ru.blackfan.bfscan.parsing.httprequests.processors;

import jadx.api.plugins.input.data.annotations.EncodedValue;
import jadx.api.plugins.input.data.annotations.IAnnotation;
import jadx.api.plugins.input.data.attributes.JadxAttrType;
import jadx.api.plugins.input.data.attributes.types.AnnotationsAttr;
import jadx.api.plugins.input.data.ILocalVar;
import jadx.core.dex.instructions.args.ArgType;
import jadx.core.dex.nodes.ClassNode;
import jadx.core.dex.nodes.FieldNode;
import jadx.core.dex.nodes.RootNode;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.time.format.DateTimeFormatter;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.blackfan.bfscan.helpers.Helpers;
import ru.blackfan.bfscan.parsing.httprequests.MultiHTTPRequest;
import ru.blackfan.bfscan.parsing.httprequests.ParameterInfo;
import ru.blackfan.bfscan.parsing.httprequests.requestbody.PrimitiveRequestBody;

public class AnnotationUtils {

    private static final Logger logger = LoggerFactory.getLogger(AnnotationUtils.class);

    public static EncodedValue getValue(Map<String, EncodedValue> values, List<String> names) {
        for (String name : names) {
            if (values.get(name) != null) {
                return values.get(name);
            }
        }
        return null;
    }

    public static String getParamName(EncodedValue fromAnnotation, ParameterInfo paramInfo, List<ILocalVar> localVars, int regNum) {
        if(paramInfo != null) {
            if(paramInfo.getName() != null) {
                return paramInfo.getName();
            }
        }
        if (fromAnnotation != null) {
            String name = Helpers.stringWrapper(fromAnnotation);
            if(paramInfo != null) {
                paramInfo.setName(name);
            }
            return name;
        }
        if (localVars != null) {
            for (ILocalVar localVar : localVars) {
                if (localVar.getRegNum() == regNum) {
                    return localVar.getName();
                }
            }
        }
        return null;
    }

    public static boolean isMultipartObject(String className) {
        return GeneralTypesConstants.MULTIPART_CLASSES.contains(className);
    }

    public static boolean isJsonObject(String className) {
        return GeneralTypesConstants.JSON_OBJECT_CLASSES.contains(className);
    }

    public static boolean isJsonArray(String className) {
        return GeneralTypesConstants.JSON_ARRAY_CLASSES.contains(className);
    }

    public static boolean isDate(String className) {
        return GeneralTypesConstants.DATE_CLASSES.contains(className);
    }

    public static boolean isArray(String className) {
        return GeneralTypesConstants.ARRAY_CLASSES.contains(className);
    }

    public static boolean isNumber(String className) {
        return GeneralTypesConstants.NUMBER_CLASSES.contains(className);
    }

    public static boolean isSimpleObject(ArgType argType) {
        return argType.isPrimitive()
                || argType.isArray()
                || argType.getObject().equals(GeneralTypesConstants.JAVA_BOOLEAN)
                || argType.getObject().equals(GeneralTypesConstants.JAVA_STRING)
                || isArray(argType.getObject())
                || isNumber(argType.getObject())
                || isDate(argType.getObject());
    }

    public static Object argTypeToValue(String fieldName, ArgType argType, RootNode rootNode, Set<String> processedClasses, boolean resolveParentClass) {
        if (argType.isPrimitive()) {
            return primitiveTypeToValue("value", argType.getPrimitiveType().getLongName());
        } else if (argType.isArray()) {
            List parameters = new ArrayList();
            parameters.add(argTypeToValue("value", argType.getArrayElement(), rootNode, new HashSet<>(processedClasses), resolveParentClass));
            return parameters;
        } else if (isArray(argType.getObject())) {
            if (argType.getGenericTypes() != null && !argType.getGenericTypes().isEmpty()) {
                List parameters = new ArrayList();
                parameters.add(argTypeToValue("value", argType.getGenericTypes().get(0), rootNode, new HashSet<>(processedClasses), resolveParentClass));
                return parameters;
            }
            return new ArrayList();
        } else if (argType.getObject().equals(GeneralTypesConstants.JAVA_OPTIONAL)) {
            if (argType.getGenericTypes() != null && !argType.getGenericTypes().isEmpty()) {
                return argTypeToValue("value", argType.getGenericTypes().get(0), rootNode, new HashSet<>(processedClasses), resolveParentClass);
            }
            return fieldName;
        } else if (argType.getObject().equals(GeneralTypesConstants.JAVA_MAP)) {
            if ((argType.getGenericTypes() != null) && !argType.getGenericTypes().isEmpty() && (argType.getGenericTypes().size() == 2)) {
                ArgType keyArg = argType.getGenericTypes().get(0);
                ArgType valueArg = argType.getGenericTypes().get(1);
                Object key = argTypeToValue("key", keyArg, rootNode, new HashSet<>(processedClasses), resolveParentClass);
                Object value = argTypeToValue("value", valueArg, rootNode, new HashSet<>(processedClasses), resolveParentClass);
                if (key != null && value != null) {
                    HashMap returnMap = new HashMap();
                    returnMap.put(key, value);
                    return returnMap;
                }
            }
            return new HashMap();
        } else {
            return classNameToDefaultValue(fieldName, argType.getObject(), rootNode, processedClasses, resolveParentClass);
        }
    }

    public static Object classNameToDefaultValue(String fieldName, String className, RootNode rootNode, Set<String> processedClasses, boolean resolveParentClass) {
        if (isJsonObject(className)) {
            return new HashMap();
        } else if (isJsonArray(className)) {
            return new ArrayList();
        } else if (isNumber(className)) {
            return 1;
        } else if (isDate(className)) {
            return LocalDate.now().format(DateTimeFormatter.ofPattern("dd-MM-yyyy"));
        } else if (className.equals(GeneralTypesConstants.JAVA_BOOLEAN)) {
            return true;
        } else if (className.equals(GeneralTypesConstants.JAVA_STRING)) {
            return fieldName;
        } else {
            ClassNode innerClass = Helpers.loadClass(rootNode, className);
            if (innerClass != null) {
                if (innerClass.isEnum()) {
                    List<FieldNode> innerClassFields = innerClass.getFields();
                    if (innerClassFields != null && (!innerClassFields.isEmpty())) {
                        return innerClassFields.get(0).getName();
                    }
                } else {
                    return classToRequestParameters(innerClass, processedClasses, resolveParentClass, rootNode);
                }
            }
            return new HashMap();
        }
    }

    public static Object primitiveTypeToValue(String fieldName, String longName) {
        return switch (longName) {
            case "boolean" ->
                true;
            case "char" ->
                fieldName;
            case "byte", "short", "int", "float", "long", "double" ->
                1;
            case "OBJECT" ->
                new HashMap();
            case "ARRAY" ->
                new ArrayList();
            default ->
                fieldName;
        };
    }

    public static void appendParametersToRequest(MultiHTTPRequest multiRequest, Map<String, Object> parameters) throws Exception {
        if (parameters != null) {
            HashMap<String, Object> bodyParameters = new HashMap();
            for (Map.Entry<String, Object> param : parameters.entrySet()) {
                Object paramValue = param.getValue();
                if (paramValue instanceof String || paramValue instanceof Number
                        || paramValue instanceof Boolean || paramValue instanceof Character) {
                    multiRequest.putQueryParameter(param.getKey(), String.valueOf(paramValue));
                } else if (paramValue instanceof List || paramValue instanceof Set) {
                    multiRequest.putQueryParameter(param.getKey(), arrayToQuery(paramValue, param.getKey()));
                } else {
                    bodyParameters.put(param.getKey(), paramValue);
                }
            }
            if (!bodyParameters.isEmpty()) {
                multiRequest.putBodyParameters(bodyParameters);
            }
        }
    }

    public static String arrayToQuery(Object list, String defaultValue) {
        if (list == null) {
            return defaultValue;
        }
        
        List<?> array;
        if (list instanceof Set) {
            array = new ArrayList<>((Set<?>) list);
        } else if (list instanceof List) {
            array = (List<?>) list;
        } else {
            return defaultValue;
        }
        
        if (array.isEmpty()) {
            return defaultValue;
        }
        
        StringBuilder result = new StringBuilder();
        boolean first = true;
        for (Object item : array) {
            if (item != null) {
                if (!first) {
                    result.append(',');
                }
                result.append(item);
                first = false;
            }
        }
        
        return URLEncoder.encode(result.toString(), StandardCharsets.UTF_8);
    }

    public static void processBodyParameter(MultiHTTPRequest request, String paramName, String defaultValue, ArgType argType, RootNode rootNode) throws Exception {
        if (paramName != null) {
            if (!defaultValue.isEmpty()) {
                request.putBodyParameter(paramName, defaultValue);
            } else {
                request.putBodyParameter(paramName, AnnotationUtils.argTypeToValue(paramName, argType, rootNode, new HashSet<>(), true));
            }
        }
    }

    public static void processArbitraryBodyParameter(MultiHTTPRequest request, String paramName, ArgType argType, RootNode rootNode) {
        try {
            if (isSimpleObject(argType)) {
                request.setRequestBody(new PrimitiveRequestBody(argTypeToValue("value", argType, rootNode, new HashSet<>(), true)));
            } else if (isMultipartObject(argType.getObject())) {
                request.setEncType("multipart");
                request.putBodyParameter(paramName != null ? paramName : "multipartName", "multipartValue");
            } else if (isJsonObject(argType.getObject())) {
                request.putBodyParameter(paramName != null ? paramName : "json", new HashMap());
            } else if (isJsonArray(argType.getObject())) {
                request.putBodyParameter(paramName != null ? paramName : "json", new ArrayList());
            } else if (isDate(argType.getObject())) {
                request.putBodyParameter(paramName != null ? paramName : "date", new ArrayList());
            } else {
                Object value = argTypeToValue("value", argType, rootNode, new HashSet<>(), true);
                if(value instanceof Map) {
                    request.putBodyParameters((HashMap)value);
                } else {
                    logger.error("Error in processArbitraryBodyParameter " + request.getClassName() + "->" + request.getMethodName());
                }
            }
        } catch (Exception ex) {
            logger.error("Error in processArbitraryBodyParameter", ex);
        }
    }

    public static void processCookieParameter(MultiHTTPRequest request, String paramName, String defaultValue) {
        if (paramName != null) {
            if (!defaultValue.isEmpty()) {
                request.putCookieParameter(paramName, defaultValue);
            } else {
                request.putCookieParameter(paramName, paramName);
            }
        }
    }

    public static void processQueryParameter(MultiHTTPRequest request, String paramName, String defaultValue, ArgType argType, RootNode rootNode) {
        if (paramName != null) {
            if (!defaultValue.isEmpty()) {
                request.putQueryParameter(paramName, defaultValue);
            } else if (isSimpleObject(argType)) {
                String value = String.valueOf(argTypeToValue("value", argType, rootNode, new HashSet<>(), true));
                if (!value.isEmpty()) {
                    request.putQueryParameter(paramName, value);
                } else {
                    request.putQueryParameter(paramName, paramName);
                }
            } else {
                request.putQueryParameter(paramName, paramName);
            }
        }
    }

    public static void processHeader(MultiHTTPRequest request, String paramName, String defaultValue) {
        if (paramName != null) {
            if (!defaultValue.isEmpty()) {
                request.putHeader(paramName, defaultValue);
            } else {
                request.putHeader(paramName, paramName);
            }
        }
    }

    public static void processPart(MultiHTTPRequest request, String paramName) throws Exception {
        request.setEncType("multipart");
        if (paramName != null) {
            request.putBodyParameter(paramName, paramName);
        } else {
            request.putBodyParameter("partKey", "partValue");
        }
    }

    public static void processHeadersFromList(MultiHTTPRequest request, ArrayList<EncodedValue> headers) {
        for (EncodedValue header : headers) {
            String[] parts = Helpers.stringWrapper(header).split(":", 2);
            if (parts.length == 2) {
                request.putHeader(parts[0].trim(), parts[1].trim());
            }
        }
    }

    public static void processPathsFromList(MultiHTTPRequest request, ArrayList<EncodedValue> paths) {
        List<String> controllerPaths = new ArrayList();
        for (EncodedValue path : paths) {
            controllerPaths.add(Helpers.stringWrapper(path));
        }
        request.setPaths(controllerPaths);
    }

    public static void processContentTypeFromList(MultiHTTPRequest request, ArrayList<EncodedValue> types) {
        if (types != null && !types.isEmpty()) {
            String selectedContentType = null;
            for (EncodedValue type : types) {
                String contentType = Helpers.stringWrapper(type).trim().toLowerCase();
                if (MultiHTTPRequest.CONTENT_TYPE_TO_ENC_TYPE.containsKey(contentType)) {
                    selectedContentType = contentType;
                    break;
                }
            }
            
            if (selectedContentType != null) {
                request.putHeader("Content-Type", selectedContentType);
            } else {
                request.putHeader("Content-Type", Helpers.stringWrapper(types.get(0)));
            }
        }
    }

    public static Map<String, Object> classToRequestParameters(ClassNode classNode, boolean resolveParentClass, RootNode rootNode) {
        return classToRequestParameters(classNode, new HashSet<>(), resolveParentClass, rootNode);
    }

    public static Map<String, Object> classToRequestParameters(ClassNode classNode, Set<String> processedClasses, boolean resolveParentClass, RootNode rootNode) {
        Map<String, Object> parameters = new HashMap<>();
        String className = classNode.getClassInfo().getFullName();
        if (!processedClasses.add(className)) {
            return parameters;
        }

        processParentClass(classNode, processedClasses, resolveParentClass, parameters, rootNode);
        processClassFields(classNode, processedClasses, resolveParentClass, parameters, rootNode);

        return parameters;
    }

    private static void processParentClass(ClassNode classNode, Set<String> processedClasses,
            boolean resolveParentClass, Map<String, Object> parameters, RootNode rootNode) {
        ArgType superClass = classNode.getSuperClass();
        if (superClass != null && resolveParentClass) {
            
            ClassNode superClassNode = Helpers.loadClass(rootNode, superClass.getObject());
            if (superClassNode != null) {
                parameters.putAll((HashMap) classToRequestParameters(superClassNode,
                        new HashSet<>(processedClasses), resolveParentClass, rootNode));
            }
        }
    }

    private static void processClassFields(ClassNode classNode, Set<String> processedClasses,
            boolean resolveParentClass, Map<String, Object> parameters, RootNode rootNode) {
        for (FieldNode field : classNode.getFields()) {
            if (isProcessableField(field)) {
                processField(field, classNode, processedClasses, resolveParentClass, parameters);
            }
        }
    }

    private static boolean isProcessableField(FieldNode field) {
        return !field.getAccessFlags().isStatic()
                && !field.getAccessFlags().isTransient()
                && !field.getAccessFlags().isSynthetic();
    }

    private static void processField(FieldNode field, ClassNode classNode,
            Set<String> processedClasses,
            boolean resolveParentClass,
            Map<String, Object> parameters) {
        String fieldName = field.getAlias();
        String defaultValue = null;

        AnnotationsAttr aList = field.get(JadxAttrType.ANNOTATION_LIST);
        if (aList != null && !aList.isEmpty()) {
            FieldAnnotationProcessor processor = new FieldAnnotationProcessor(classNode.root());
            processor.process(aList, fieldName, defaultValue);
            fieldName = processor.getFieldName();
            defaultValue = processor.getDefaultValue();
        }

        if (defaultValue != null) {
            parameters.put(fieldName, defaultValue);
        } else {
            parameters.put(fieldName, argTypeToValue(fieldName, field.getType(),
                    classNode.root(), new HashSet<>(processedClasses), resolveParentClass));
        }
    }

    public static String getAnnotationClass(IAnnotation ann, RootNode root) {
        String aClsAlias = Helpers.getJvmAlias(root, ann.getAnnotationClass());
        if (aClsAlias != null) {
            return aClsAlias;
        }
        return ann.getAnnotationClass();
    }
}
